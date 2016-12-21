#include <ax.hpp>

#include <cmath>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <vector>

#include <ax.mpmcsplf.hpp>

using namespace ax;

struct counter {
    static size_t ctor_counter;
    static size_t dtor_counter;
    
    counter(int)  { ++ctor_counter; }
    ~counter()    { ++dtor_counter; }
};
size_t counter::ctor_counter = 0;
size_t counter::dtor_counter = 0;

template <typename T>
using queue_template = mpmcsplf::queue<T, 4_MIB>;

void queue_test() {
    
    /// Single thread
    for(const size_t N : {1_KIB, 32_KIB, 64_KIB}) {
        using q_t = queue_template<size_t>;
        q_t q;
        
        LIGHT_TEST(!q.extract());
        
        for(size_t i = 0; i < N; ++i)
            q.emplace(i*2);
        
        for(size_t i = 0; i < N; ++i) {
            auto ex = q.extract();
            *ex /= 2;
            LIGHT_TEST(*ex == i);
        }
        
        LIGHT_TEST(!q.extract());
    }
    
    /// C-tor/d-tor
    {
        using q_t = queue_template<counter>;
        q_t q;
        
        const size_t N = 1_KIB;
        for(size_t i = 0; i < N; ++i)
            q.emplace(i);
        
        LIGHT_TEST(counter::ctor_counter == N);
        
        while(q.extract());
        
        LIGHT_TEST(counter::dtor_counter == N);
    }
    
    LIGHT_TEST(counter::ctor_counter == counter::dtor_counter);
    
    /// Multithread
    {
        using q_t = mpmcsplf::queue<void, 64_KIB>;
        
        q_t q;
        
        const size_t N = 1_KIB;
        
        struct id_num {
            size_t id;
            size_t num;
        };
        
        struct little_one {
            id_num x;
        };
        
        struct medium_one {
            union {
                id_num x;
                std::array<aligned_cacheline, 4> arr_;
            };
        };
        
        struct gigant_one {
            union {
                id_num x;
                std::array<aligned_cacheline, 7> arr_;
            };
        };
        
        volatile bool start = false;
        const size_t hardware_concurrency = std::thread::hardware_concurrency();
        const size_t N_threads = hardware_concurrency > 1 ? (hardware_concurrency - 1)/2 + 1 : 4;
        
        std::atomic<size_t> writers_finished = ATOMIC_VAR_INIT(0);
        std::map<size_t, std::set<size_t>> result;
        std::mutex map_lock;
        
        std::vector<std::thread> threads;
        
        /// Writers
        for(size_t id = 0; id < N_threads; ++id)
            threads.emplace_back([&](size_t id){
                while(!start) {}
                for(size_t i = 0; i < N; ++i) {
                    switch(i % 3) {
                        case 0: q.emplace<little_one>(little_one{{id, i}}); break;
                        case 1: q.emplace<medium_one>(medium_one{{id, i}}); break;
                        case 2: q.emplace<gigant_one>(gigant_one{{id, i}}); break;
                    }
                }
                ++writers_finished;
            }, id);
        
        /// Readers
        for(size_t id = 0; id < N_threads; ++id)
            threads.emplace_back([&](size_t id){
                while(!start) {}
                
                q_t::extracted_type<id_num> ex;
                
                while(true) {
                    auto finished = writers_finished.load();
                    if(ex = q.extract<id_num>()) {
                        auto value = *ex;
                        
                        if(std::rand() % 2 == 0)
                            ex.left();
                        
                        std::lock_guard<std::mutex> guard(map_lock);
                        auto& set_by_id = result[value.id];
                        auto inserted = set_by_id.emplace(value.num);
                        LIGHT_TEST(inserted.second);
                        
                    } else if(finished == N_threads)
                        break;
                }
            }, id);
        
        start = true;
        
        for(auto& t : threads)
            t.join();
        
        std::set<size_t> stardard_set;
        for(size_t i = 0; i < N; ++i)
            stardard_set.emplace(i);
        
        for(auto const& pair : result)
            LIGHT_TEST(stardard_set == pair.second);
        
        LIGHT_TEST(!q.extract());
    }
}

int main() {
    queue_test();
}