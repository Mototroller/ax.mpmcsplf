#pragma once

#include <ax.hpp>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <thread>

#define LOG_HEAD "[mpmcsplf]: "

namespace ax { namespace mpmcsplf {
    
    /**
     * Implementation of MPMCSPLF, which is
     * Multi Producer Multi Consumer Single Process Lock Free queue.
     * @arg V - stored type. "void" for custom memory management
     * @arg Page_size - size of preallocated raw storages
     * @arg Allocator - inner nodes allocator
     * TODO: exeptions handling
     */
    template <
        typename V,
        size_t Page_size,
        template <typename> class Allocator = huge_space_allocator
    >
    class queue {
    private:
        
        using flags_t   = uint16_t;
        using page_id_t = uint32_t;
        using offset_t  = uint32_t;
        
        enum COMMIT_STATUS : flags_t {
            EMPTY       = 0,
            FILLING     = 1,
            COMMITED    = 2
        };
        
        enum PAGE_SWITCH_STATUS : flags_t {
            NORMAL = 0,
            SWITCH = 1
        };
        
        /// POD status type to be atomic in a slot's header
        struct status_t {
            COMMIT_STATUS       status;
            PAGE_SWITCH_STATUS  page_switch;
            offset_t            indexes;
            
            constexpr static status_t EMPTY_STATUS() {
                return status_t{EMPTY, NORMAL, 0}; }
        };
        
        /// Page with cachelines as slots
        struct page_t {
            struct slot_t {
                
                /// POD header
                struct slot_header_t {
                    /// Commit state + size in indexes
                    std::atomic<status_t> status;
                    
                    /// Requested size in bytes, valid after commiting
                    size_t requested_size;
                    
                    /*
                     * To be true-POD-type it shouldn't contain any special functions.
                     * NOTE: special allocator requirements listed below
                     * 
                     * If it will be implemented in the future, it should be like:
                     * 
                     * // See: constexpr atomic( T desired );
                     * constexpr slot_header_t() : status(status_t::EMPTY_STATUS()) {}
                     */
                };
                
                // --- Data fields --- //
                
                union {
                    slot_header_t header;
                    aligned_cacheline pad_[1];
                    // stored data here [1+]
                };
                
                void const* data() const {
                    return static_cast<void const*>(&pad_[1]); }
                
                void* data() {
                    return const_cast<void*>(static_cast<slot_t const*>(this)->data()); }
            };
            
            enum : size_t { size_in_bytes   = Page_size };
            enum : size_t { slot_size       = sizeof(slot_t) };
            enum : size_t { slots_amount    = size_in_bytes/slot_size };
            enum : size_t { slots_per_page  = slots_amount };
            
            /// Page slots (data fields)
            union {
                slot_t slots[1];
                aligned_cacheline pad_[slots_per_page];
                // real slots array here [0+]
            };
            
            // constexpr page_t() : slots() {};
        };
        
        using page_type = page_t;
        using slot_type = typename page_type::slot_t;
        
        /**
         * Entity containing allocated page and necessary info for concurrent access.
         * NOTE: allocator must be zero-initializing (TODO custom one)
         */
        struct node {
            page_type           page;       // POD, uninitialized (see allocator requirements)
            page_id_t           page_id;    // uinique incremental page ID
            std::atomic<node*>  prev;       // pointer to previous node
            std::atomic<size_t> internal_counter;
            
            node(node* prev_node, page_id_t id) :
                // page(), // POD
                page_id(id),
                prev(prev_node),
                internal_counter(0) {}
        };
        
        using node_allocator = Allocator<node>;
        struct node_deleter { void operator()(node* ptr) { node_allocator{}.deallocate(ptr, 1); } };
        
        /// 64-bit struct ~ slot coordinate
        typedef struct id_offset {
            page_id_t id;       // page ID
            offset_t offset;    // writer - first free, reader - first unread slot
        } coordinate;
        
        /**
         * RAII object, acquires/releases node's internal counter in c-tor/d-tor.
         * Movable, nullable.
         */
        struct node_holder {
        private:
            node* node_;
            
            static inline void acquire(node* ptr) {
                if(ptr) ptr->internal_counter.fetch_add(1, std::memory_order_release); }
            
            static inline void release(node* ptr) {
                if(ptr) ptr->internal_counter.fetch_sub(1, std::memory_order_release); }
            
        public:
            explicit node_holder(node* ptr) : node_(ptr) {
                acquire(node_); }
            
            ~node_holder() {
                release(node_); }
            
            node_holder(node_holder&& other) noexcept :
                node_(other.node_) {
                other.node_ = nullptr;
            }
            
            friend void swap(node_holder& lh, node_holder& rh) noexcept {
                using std::swap;
                swap(lh.node_, rh.node_);
            }
            
            node_holder& operator=(node_holder other) noexcept {
                swap(*this, other);
                return *this;
            }
            
            node* get_node() const {
                return node_; }
            
            /**
             * Releases ownership, can be used for acquiring node with id > prev
             * or while moving assignment (acquire(new) -> release(old))
             */
            void reset(node* new_node) {
                acquire(new_node);
                release(node_);
                node_ = new_node;
            }
        };
        
        /// Thread-local cache and info
        struct local_data {
            using reserved_node_type = std::unique_ptr<node, node_deleter>;
            reserved_node_type reserved_node;
        };
        
        static local_data& get_thread_local_data() {
            static thread_local local_data local_;
            return local_;
        }
        
        /**
         * RAII-object, contains node_holder and pointer to allocated space,
         * commits it implicitly or in d-tor (if user didn't do it explicitly).
         * TODO: exceptions handling and storing
         */
        template <typename T>
        struct allocated {
        private:
            friend class queue;
            
            node_holder holder_;
            slot_type*  slot_;
            
            /// Can be constructed only by queue
            explicit allocated(node_holder&& holder, slot_type* slot) :
                holder_ (std::move(holder)),
                slot_   (slot) {}
            
            friend void swap(allocated& lh, allocated& rh) noexcept {
                using std::swap;
                swap(lh.holder_, rh.holder_);
                swap(lh.slot_,   rh.slot_);
            }
        
        public:
            
            /// Default c-tor, empty entity
            allocated() : allocated(node_holder(nullptr), nullptr) {}
            
            /// Copy disabled
            allocated(allocated const&) = delete;
            
            /// Moving is allowed
            allocated(allocated&& other) noexcept :
                holder_ (std::move(other.holder_)),
                slot_   (other.slot_) {
                other.slot_ = nullptr;
            }
            
            /// Unified assignment
            allocated& operator=(allocated other) noexcept {
                swap(*this, other);
                return *this;
            }
            
            /// Explicitly sets status to COMMITED
            void commit() {
                if(slot_) {
                    auto indexes = slot_->header.status.load(std::memory_order_relaxed).indexes;
                    auto prev_status = slot_->header.status.exchange({COMMITED, NORMAL, indexes}, std::memory_order_release);
                    assert(prev_status.status == FILLING); // avoidind double commiting and sequence errors
                    assert(prev_status.indexes == indexes); // test for relaxed order
                    holder_.reset(nullptr);
                    slot_ = nullptr;
                }
            }
            
            ~allocated() {
                commit(); }
            
            size_t size() const {
                return slot_->header.requested_size; }
            
            /// @returns casted pointer to allocated space
            T* ptr() const {
                return static_cast<T*>(slot_->data()); }
        };
        
        /**
         * RAII iterator-like object, contains pointer to extracted data,
         * releases slot ownership in d-tor.
         * TODO: exceptions handling and storing
         */
        template <typename T>
        struct extracted {
        private:
            friend class queue;
            
            queue*      this_;
            node_holder holder_;
            slot_type*  slot_;
            
            /// Can be constructed only by queue
            explicit extracted(queue* that, node_holder&& holder, slot_type* slot) :
                this_   (that),
                holder_ (std::move(holder)),
                slot_   (slot) {}
            
            template <typename U = T>
            inline typename std::enable_if<std::is_destructible<U>::value>::type
            destroy_slot() const {
                cptr()->~T(); }
            
            template <typename U = T>
            inline typename std::enable_if<!std::is_destructible<U>::value>::type
            destroy_slot() const {}
            
            friend void swap(extracted& lh, extracted& rh) noexcept {
                using std::swap;
                swap(lh.this_,   rh.this_);
                swap(lh.holder_, rh.holder_);
                swap(lh.slot_,   rh.slot_);
            }
        
        public:
            
            extracted() : extracted(nullptr, node_holder(nullptr), nullptr) {}
            
            /// Copy disabled
            extracted(extracted const&) = delete;
            
            /// Moving is allowed
            extracted(extracted&& other) noexcept :
                this_   (other.this_),
                holder_ (std::move(other.holder_)),
                slot_   (other.slot_) {
                other.this_ = nullptr;
                other.slot_ = nullptr;
            }
            
            /// Unified assignment
            extracted& operator=(extracted other) noexcept {
                swap(*this, other);
                return *this;
            }
            
            /// Releases ownership, destroys stored object
            void left() {
                if(slot_) {
                    destroy_slot();
                    
                    // Here object d-tor was called and slot_ will be set to nullptr.
                    // There are no references to data left, we can try cleaning.
                    
                    node* my_node = holder_.get_node();
                    
                    // We are the last thread having access to this node
                    if(my_node->internal_counter.load(std::memory_order_acquire) == 1) {
                        size_t tail_left_ids_less_than = this_->tail_left_ids_less_than_.load(std::memory_order_acquire);
                        size_t head_left_ids_less_than = this_->head_left_ids_less_than_.load(std::memory_order_acquire);
                        size_t min_reachable_id = std::min(tail_left_ids_less_than, head_left_ids_less_than);
                        
                        // --- [] [a] [] [bc] [] [x] [] [d] | [MR] [] --> HEAD
                        if(unlikely(my_node->page_id < min_reachable_id)) {
                            
                            // "Double check"
                            bool fact_tail = (my_node->internal_counter.load() == 1);
                            if(fact_tail) {
                                node* p = my_node;
                                while((p = p->prev.load(std::memory_order_acquire))) {
                                    if(p->internal_counter.load(std::memory_order_acquire) > 0) {
                                        fact_tail = false;
                                        break;
                                    }
                                }
                            }
                            
                            // There aren't visitors deeper to the tail
                            if(fact_tail) {
                                node* old_prev = my_node->prev.exchange(nullptr);
                                destroy_node_and_deeper(old_prev);
                                
                                // TODO: one unused left
                                // TODO: check sequience
                            }
                        }
                    }
                    this_ = nullptr;
                    holder_.reset(nullptr);
                    slot_ = nullptr;
                }
            }
            
            operator bool() const {
                return slot_ != nullptr; }
                
            T* operator->() const {
                return ptr(); }
            
            template <typename U = T>
            typename std::enable_if<!std::is_void<U>::value, U>::type&
            operator*() const {
                return *(ptr()); }
            
            ~extracted() {
                left(); }
            
            /// @returns casted pointer to extracted space
            T* ptr() const {
                return static_cast<T*>(slot_->data()); }
            
            T const* cptr() const {
                return static_cast<T const*>(ptr()); }
        };
        
        // --- Data fields --- //
        
        union {
            /// Pointer to last allocated node (so and page), writers endpoint
            std::atomic<node*> head_node_;
            aligned_cacheline UNIQUENAME;
        };
        
        union {
            /// Current head page's id and offset of empty slot
            std::atomic<coordinate> head_coordinate_;
            aligned_cacheline UNIQUENAME;
        };
        
        union {
            /// External counter for writers
            std::atomic<size_t> allocations_in_progress_;
            aligned_cacheline UNIQUENAME;
        };
        
        union {
            /// Updates every holder's release, shows unreachable pages IDs
            std::atomic<size_t> head_left_ids_less_than_;
            aligned_cacheline UNIQUENAME;
        };
        
        union {
            std::atomic<coordinate> tail_coordinate_;
            aligned_cacheline UNIQUENAME;
        };
        
        union {
            /// External counter for readers
            std::atomic<size_t> extractings_in_progress_;
            aligned_cacheline UNIQUENAME;
        };
        
        union {
            std::atomic<size_t> tail_left_ids_less_than_;
            aligned_cacheline UNIQUENAME;
        };
        
        std::atomic<node*> reserved_node_;
        node_allocator allocator_;
        
        /**
         * Same as:
         * what.CAS(expected <=> new_id_offset)
         * what.compare_exchange_strong(expected, desired, std::memory_order_acq_rel);
         */
        template <typename T, typename E, typename D>
        inline static bool strong_CAS(T& what, E& expected, D const& desired) {
            return what.compare_exchange_strong(
                expected,
                desired,
                std::memory_order_acq_rel,
                std::memory_order_acquire);
        }
        
        /**
         * @arg head_id - ID of guaranteed acquired node (will be analyzed)
         * @returns pointer to found node, nullptr otherwise
         */
        node* find_node_by_id_not_less_than(page_id_t id, page_id_t head_id) {
            node* found = head_node_.load(std::memory_order_acquire);
            while(found->page_id != id) {
                if(found->page_id > head_id)
                    found = found->prev.load(std::memory_order_acquire);
                else
                    return nullptr;
            }
            return found;
        }
        
        /// @returns holder with found existing (!) node
        node_holder construct_holder_by_id(page_id_t id) {
            node* found = head_node_.load(std::memory_order_acquire);
            while(found->page_id != id)
                found = found->prev.load(std::memory_order_acquire);
            return node_holder(found);
        }
        
        /**
         * 1. Increments external counter for the endpoint (acquires "cleaning lock").
         * 2. Constructs holder by endpoint ID (node guaranteed to be exists ^).
         * 3. Decrements external counter (releases "cleaning lock").
         * 4. If external counter EC == 0, moves respective unreachable counter to EC.
         */
        inline node_holder acquire_node(
            std::atomic<coordinate>&    node_coordinate,
            std::atomic<size_t>&        counter,
            coordinate&                 result,
            std::atomic<size_t>&        left_ids_less_than) {
            
            counter.fetch_add(1, std::memory_order_release);
            
            result = node_coordinate.load(std::memory_order_acquire);
            node_holder holder(construct_holder_by_id(result.id));
            
            auto final_external_counter = counter.fetch_sub(1, std::memory_order_release) - 1;
            
            /// This thread was the last locking cleaning
            if(final_external_counter == 0) {
                // ...[][][][][][result.id][]...
                // -------------| <- wouldn't be accessed
                size_t last_reachable_id = result.id;
                auto prev_value = left_ids_less_than.load(std::memory_order_relaxed);
                if(last_reachable_id > prev_value)
                    left_ids_less_than.store(last_reachable_id, std::memory_order_release);
            }
            
            return holder;
        }
        
        /**
         * @returns holder with head node
         * @arg head_ioc - reference for storing consistent coordinate
         */
        inline node_holder acquire_head_node(coordinate& head_io) {
            return acquire_node(
                head_coordinate_,
                allocations_in_progress_,
                head_io,
                head_left_ids_less_than_);
        }
        
        /**
         * @returns holder with tail node
         * @arg tail_ioc - reference for storing consistent coordinate
         */
        inline node_holder acquire_tail_node(coordinate& tail_io) {
            return acquire_node(
                tail_coordinate_,
                extractings_in_progress_,
                tail_io,
                tail_left_ids_less_than_);
        }
        
        /// Destroys nodes, explicitly calls operator delete() for all in the chain
        static void destroy_node_and_deeper(node* first) {
            size_t n = 0;
            while(first != nullptr) {
                node* next = first->prev.load(std::memory_order_acquire);
                node_allocator{}.deallocate(first, 1);
                first = next;
                ++n;
            }
        }
        
        /**
         * Tries to move head_node_ pointer to reserved_node_ or given local reserved (dummy).
         * @arg dummy - possible candidate to be mounted as head (nullable, can be changed)
         * @arg current_head_holder - reference to head RAII-holder
         */
        bool reserved_or_dummy_to_head(node*& dummy, node_holder const& current_head_holder) {
            if(dummy || (dummy = reserved_node_.exchange(dummy, std::memory_order_acq_rel))) {
                /// Here dummy is not-null and points to valid node
            
                node* current_head = current_head_holder.get_node();
                dummy->prev = current_head;
                dummy->page_id = current_head->page_id + 1;
                
                if(strong_CAS(head_node_, current_head, dummy)) {
                    dummy = nullptr;
                    return true;
                }
            }
            return false;
        }
        
        /// Uses existing reserved node or allocates new to be mounted as head
        void mount_new_head(node_holder const& current_head_holder) {
            auto& local = get_thread_local_data();
            node* dummy = local.reserved_node.release();
            if(reserved_or_dummy_to_head(dummy, current_head_holder)) {
                /// Reserved node was successfuly set
            } else if(!dummy) {
                /// Allocation needed
                dummy = allocator_.allocate(1);
                new(dummy) node(nullptr, 0);
            }
            local.reserved_node.reset(dummy);
        }
        
        /// Restores main shared reserved_node_ with local or newly created
        void restore_main_reserved() {
            auto& local = get_thread_local_data();
            node* current_reserved = reserved_node_.load(std::memory_order_relaxed);
            if(!current_reserved) {
                node* dummy = local.reserved_node.release();
                if(!dummy) {
                    dummy = allocator_.allocate(1);
                    new(dummy) node(nullptr, 0);
                }
                dummy = reserved_node_.exchange(dummy, std::memory_order_release);
                local.reserved_node.reset(dummy);
            }
        }
        
    public:
        
        template <typename T>
        using allocated_type = allocated<T>;
        
        template <typename T>
        using extracted_type = extracted<T>;
        
        queue() :
            head_coordinate_({0}),
            allocations_in_progress_(0),
            head_left_ids_less_than_(0),
            
            tail_coordinate_(head_coordinate_.load()),
            extractings_in_progress_(0),
            tail_left_ids_less_than_(0) {
            
            head_node_     = new(allocator_.allocate(1)) node(nullptr, 0);
            reserved_node_ = new(allocator_.allocate(1)) node(nullptr, 0);
        }
        
        /// Copy and move disabled
        queue(queue const&)            = delete;
        queue& operator=(queue const&) = delete;
        
        ~queue() {
            while(extract());
            destroy_node_and_deeper(head_node_.load());
            if(reserved_node_)
                allocator_.deallocate(reserved_node_.load(), 1);
        }
        
        std::string dump() const {
            return strprintf("{HEAD %% {id:%% off:%%} alls:%% left<%% | TAIL {id:%% off:%%} exts:%% left<%%}",
                head_node_.load(),
                head_coordinate_.load().id,
                head_coordinate_.load().offset,
                allocations_in_progress_.load(),
                head_left_ids_less_than_.load(),
                
                tail_coordinate_.load().id,
                tail_coordinate_.load().offset,
                extractings_in_progress_.load(),
                tail_left_ids_less_than_.load());
        }
        
        /**
         * Allocates requested size and
         * @returns allocated RAII object.
         */
        template<typename T = V>
        allocated<T> allocate(size_t size = sizeof(T));
        
        /// Constructs object in-place using placement new()
        template <typename T = V, typename... Args>
        void emplace(Args&&... args) {
            auto a = allocate<T>();
            new(a.ptr()) T(std::forward<Args>(args)...); }
        
        /// @returns unique_ptr pointing to moved extracted object [[deprecated("use extract()")]]
        template <typename T = V>
        std::unique_ptr<T> pop() {
            auto ex = extract<T>();
            auto obj = ex ? new T(std::move(*ex)) : nullptr;
            return std::unique_ptr<T>(obj);
        }
        
        /// @returns RAII object holding the extracted slot
        template <typename T = V>
        extracted<T> extract();
    };
    
} // mpmcsplf
} // ax

#include <ax.mpmcsplf_impl.hpp>

#undef LOG_HEAD
