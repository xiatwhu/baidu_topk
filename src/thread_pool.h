#pragma once

#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <vector>
#include <cassert>

inline int do_some_nops() {
    asm volatile (
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
        "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n" "nop\n"
    );
    return 64;
}

constexpr int max_busy_wait_nops = 4 * 1000 * 1000;

template <typename T>
T wait_for_change(
        std::atomic<T>* var,
        T init_value,
        std::condition_variable* cond,
        std::mutex* mutex) {
    int nops = 0;
    T new_value = var->load(std::memory_order_acquire);
    if (new_value != init_value) {
        return new_value;
    }

    while (nops < max_busy_wait_nops) {
        nops += do_some_nops();
        new_value = var->load(std::memory_order_acquire);
        if (new_value != init_value) {
            return new_value;
        }
    }

    std::unique_lock<std::mutex> g(*mutex);
    new_value = var->load(std::memory_order_acquire);
    cond->wait(g, [&]() {
        new_value = var->load(std::memory_order_acquire);
        return new_value != init_value;
    });
    return new_value;
}

class Barrier {
public:

    Barrier() : _count(0) {}

    void reset(std::size_t count) {
        std::size_t old_count = _count.load(std::memory_order_relaxed);
        assert(old_count == 0);
        (void)old_count;
        _count.store(count, std::memory_order_release);
    }

    bool decrement() {
        std::size_t old_count_value = _count.fetch_sub(1, std::memory_order_acq_rel);
        assert(old_count_value > 0);
        std::size_t count_value = old_count_value - 1;
        return count_value == 0;
    }

    void wait() {
        int nops = 0;
        while (_count.load(std::memory_order_acquire)) {
            nops += do_some_nops();
            if (nops > max_busy_wait_nops) {
                nops = 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

private:
    std::atomic<std::size_t> _count;
};

struct Task {
    virtual void run() = 0;
    virtual ~Task() = default;
};

class Worker {
public:
    enum class State :uint8_t {
        Start,
        Ready,
        Busy,
        Exit
    };

    explicit Worker(Barrier* barrier)
        : _task(nullptr),
          _state(State::Start),
          _barrier(barrier) {
        _thread.reset(new std::thread([this]() { this->run();}));
    }

    ~Worker() {
        change_state(State::Exit);
        _thread->join();
    }

    void change_state(State new_state) {
        std::lock_guard<std::mutex> g(_state_mutex);
        assert(new_state != _state.load(std::memory_order_relaxed));
        switch (_state.load(std::memory_order_relaxed)) {
        case State::Start:
            assert(new_state == State::Ready);
            break;
        case State::Ready:
            assert(new_state == State::Busy || new_state == State::Exit);
            break;
        case State::Busy:
            assert(new_state == State::Ready || new_state == State::Exit);
            break;
        default:
            break;
        }
        _state.store(new_state, std::memory_order_relaxed);
        _state_cond.notify_one();
        if (new_state == State::Ready) {
            _barrier->decrement();
        }
    }

    void run() {
        change_state(State::Ready);
        while (true) {
            State new_state = wait_for_change(&_state, State::Ready, &_state_cond, &_state_mutex);
            switch (new_state) {
            case State::Busy:
                (*_task).run();
                _task = nullptr;
                change_state(State::Ready);
                break;
            case State::Exit:
                return;
            default:
                break;
            }
        }
    }

    void run_task(Task* task) {
        _task = task;
        assert(_state.load(std::memory_order_acquire) == State::Ready);
        change_state(State::Busy);
    }

private:
    std::unique_ptr<std::thread> _thread;
    Task* _task;

    std::condition_variable _state_cond;
    std::mutex _state_mutex;

    std::atomic<State> _state;
    Barrier* const _barrier;
};

class ThreadPool {
public:
    ~ThreadPool() {
        for (auto* w : _workers) {
            delete w;
        }
    }

    size_t num_threads() {
        return _workers.size() + 1;
    }

    void set_num_threads(size_t nt) {
        assert(nt > 1);
        size_t total_worker = nt - 1;
        if (_workers.size() >= total_worker) {
            return;
        }
        _barrier.reset(total_worker - _workers.size());
        while (_workers.size() < total_worker) {
            _workers.push_back(new Worker(&_barrier));
        }
        _barrier.wait();
    }

    void run_task(const std::vector<Task*>& tasks) {
        assert(tasks.size() >= 1);
        set_num_threads(tasks.size());
        std::size_t workers_count = tasks.size() - 1;
        _barrier.reset(workers_count);
        for (size_t i = 0; i < workers_count; ++i) {
            _workers[i]->run_task(tasks[i]);
        }
        Task* task = tasks.back();
        task->run();

        _barrier.wait();
        for (Task* entry : tasks) {
            delete entry;
        }
    }

private:
    std::vector<Worker*> _workers;
    Barrier _barrier;
};