#include <ax.mpmcsplf.hpp>

namespace ax { namespace mpmcsplf {
    
    template <typename V, size_t S, template <typename> class A>
    template<typename T>
    auto queue<V,S,A>::
    allocate(size_t size) -> allocated<T> {
        const size_t indexes_needed = (size - 1)/page_type::slot_size + 1 + 1;
        
        do {
            coordinate current_head_coordinate;
            
            /// RAII initializing
            node_holder current_head_holder(acquire_head_node(current_head_coordinate));
            
            /// Here current_head_coordinate is valid and points to the existing page/slot
            
            /// There is at least one free slot at the current page...
            if(likely(current_head_coordinate.offset < page_type::slots_per_page)) {
                
                node* current_node = current_head_holder.get_node();
                
                /// ... and there is enough space to store "size" bytes
                offset_t indexes_left = page_type::slots_per_page - current_head_coordinate.offset;
                bool enough_space = indexes_needed <= indexes_left;
                
                status_t expected_status = status_t::EMPTY_STATUS();
                status_t new_status = enough_space ?
                    status_t{FILLING,  NORMAL, static_cast<offset_t>(indexes_needed)}: // ordinary slot
                    status_t{COMMITED, SWITCH, static_cast<offset_t>(indexes_left)};   // fake slot -> switch page
                
                coordinate new_head_coordinate = current_head_coordinate;
                
                /// After capturing "head" += "captured", regardless of page switching
                new_head_coordinate.offset += new_status.indexes;
                
                /// All calculations completed, lets roll
                
                slot_type& slot = current_node->page.slots[current_head_coordinate.offset];
                bool captured = strong_CAS(slot.header.status, expected_status, new_status);
                
                /// Slot has been captured
                if(likely(captured)) {
                    
                    // Here head_coordinate_ cannot be switched to the next page (see condition above),
                    // so it only can be moved forward inside current page.
                    
                    strong_CAS(head_coordinate_, current_head_coordinate, new_head_coordinate);
                    
                    /// Ordinary slot, return
                    if(likely(enough_space)) {
                        slot.header.requested_size = size;
                        return allocated<T>(std::move(current_head_holder), &slot);
                        // [[fallthrough]] ===>
                    
                    /// else - page switching, restart
                    } else {
                        /// ~current_head_holder
                        continue;
                    }
                    
                /// Wasted (slot has been captured by someone else), trying to help
                } else {
                    coordinate new_head_ioc = current_head_coordinate;
                    new_head_ioc.offset += expected_status.indexes;
                    
                    if(head_coordinate_.compare_exchange_weak(
                        current_head_coordinate,
                        new_head_ioc,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire
                    )) {
                        // this thread completed others transaction
                    } else {
                        // transaction has been complited by someone else
                    }
                    
                    /// ~current_head_holder
                    continue;
                }
            
            /// Page is full, allocation needed
            } else {
                assert(current_head_coordinate.offset == page_type::slots_per_page);
                
                /**
                    * 1. Create new node
                    * 2. Update head_
                    * 3. Update head_id_offset_
                    */
                
                /// Find current head node (assumed to be real head)
                page_id_t current_id = current_head_coordinate.id;
                page_id_t new_id = current_id + 1;
                
                /// If page already exists, try to update head_coordinate_
                if(node* existing = find_node_by_id_not_less_than(new_id, current_id)) {
                    coordinate new_head_coordinate          = current_head_coordinate;
                    new_head_coordinate.id                  = new_id;
                    new_head_coordinate.offset              = 0;
                    if(strong_CAS(head_coordinate_, current_head_coordinate, new_head_coordinate)) {
                        // new page exists + updated
                    }
                    /// Just tried to help (to avoid live-lock), restart anyway
                    
                    /// ~current_head_holder
                    continue;
                }
                
                /// ... seems like there is no allocated next page
                
                mount_new_head(current_head_holder);
                
                /// ~current_head_holder
                continue;
            }
        } while(true);
        
        // <===
        throw std::bad_alloc{};
    }
    
    template <typename V, size_t S, template <typename> class A>
    template<typename T>
    auto queue<V,S,A>::
    extract() -> extracted<T> {
        do {
            coordinate current_tail_coordinate;
            node_holder current_tail_holder(acquire_tail_node(current_tail_coordinate));
            
            assert(current_tail_coordinate.offset <= page_type::slots_per_page);
            node* current_tail_node = current_tail_holder.get_node();
            
            /// There is at least one valid slot at the current reader page
            if(likely(current_tail_coordinate.offset < page_type::slots_per_page)) {
                
                slot_type& slot = current_tail_node->page.slots[current_tail_coordinate.offset];
                const status_t status = slot.header.status.load(std::memory_order_acquire);
                
                /// Slot commited and valid
                if(status.status == COMMITED) {
                    coordinate new_tail_coordinate = current_tail_coordinate;
                    new_tail_coordinate.offset += status.indexes;
                    
                    /// Slot has been captured
                    if(likely(strong_CAS(tail_coordinate_, current_tail_coordinate, new_tail_coordinate))) {
                        
                        /// Ordinary slot
                        if(likely(status.page_switch == NORMAL))
                            return extracted<T>(this, std::move(current_tail_holder), &slot);
                        
                        /// SWITCH slot, restart
                        else
                            continue;
                        
                    /// Wasted, restart
                    } else
                        continue;
                    
                /// Filling or Empty
                } else {
                    if(status.status == EMPTY)
                        restore_main_reserved();
                    return extracted<T>();
                }
            }
            
            /// else - page is full, switch
            page_id_t new_id = current_tail_coordinate.id + 1;
            
            coordinate current_head_coordinate = head_coordinate_.load(std::memory_order_acquire);
            
            /// Next page doesn't exists
            if(new_id > current_head_coordinate.id) {
                restore_main_reserved();
                return extracted<T>();
            }
            
            /// else - move coordinate
            
            coordinate new_tail_coordinate;
            new_tail_coordinate.id      = new_id;
            new_tail_coordinate.offset  = 0;
            strong_CAS(tail_coordinate_, current_tail_coordinate, new_tail_coordinate);
            
        } while(true);
    }
    
} // hl
} // ax
