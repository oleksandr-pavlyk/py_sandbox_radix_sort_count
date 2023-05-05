
#include <sycl/sycl.hpp>
#include <type_traits>
#include <cstdint>
#include <limits>

namespace radix_sort {

// rounded up result of (__number / __divisor)
template <typename _T1, typename _T2>
constexpr auto
__ceiling_div(_T1 __number, _T2 __divisor) -> decltype((__number - 1) / __divisor + 1)
{
    return (__number - 1) / __divisor + 1;
}


template <bool __is_ascending>
bool
__order_preserving_cast(bool __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return !__val;
}

template <bool __is_ascending, typename _UInt, std::enable_if_t<::std::is_unsigned_v<_UInt>, int> = 0>
_UInt
__order_preserving_cast(_UInt __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return ~__val; //bitwise inversion
}

template <bool __is_ascending, typename _Int,
          std::enable_if_t<::std::is_integral_v<_Int>&& ::std::is_signed_v<_Int>, int> = 0>
::std::make_unsigned_t<_Int>
__order_preserving_cast(_Int __val)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask = (__is_ascending) ? (_UInt(1) << ::std::numeric_limits<_Int>::digits) : (~_UInt(0) >> 1);
    return __val ^ __mask;
}

template <bool __is_ascending, typename _Float,
         std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t), int> = 0>
::std::uint32_t
__order_preserving_cast(_Float __val)
{
    ::std::uint32_t __uint32_val = sycl::bit_cast<::std::uint32_t>(__val);
    ::std::uint32_t __mask;
    // __uint32_val >> 31 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint32_val >> 31 == 0) ? 0x80000000u : 0xFFFFFFFFu;
    else
        __mask = (__uint32_val >> 31 == 0) ? 0x7FFFFFFFu : ::std::uint32_t(0);
    return __uint32_val ^ __mask;
}

template <bool __is_ascending, typename _Float,
          std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t), int> = 0>
::std::uint64_t
__order_preserving_cast(_Float __val)
{
    ::std::uint64_t __uint64_val = sycl::bit_cast<::std::uint64_t>(__val);
    ::std::uint64_t __mask;
    // __uint64_val >> 63 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint64_val >> 63 == 0) ? 0x8000000000000000u : 0xFFFFFFFFFFFFFFFFu;
    else
        __mask = (__uint64_val >> 63 == 0) ? 0x7FFFFFFFFFFFFFFFu : ::std::uint64_t(0);
    return __uint64_val ^ __mask;
}


//------------------------------------------------------------------------
// radix sort: bucket functions
//------------------------------------------------------------------------

// get number of buckets (size of radix bits) in T
template <typename _T>
constexpr ::std::uint32_t
__get_buckets_in_type(::std::uint32_t __radix_bits)
{
    return __ceiling_div(sizeof(_T) * ::std::numeric_limits<unsigned char>::digits, __radix_bits);
}

// get bits value (bucket) in a certain radix position
template <::std::uint32_t __radix_mask, typename _T>
::std::uint32_t
__get_bucket(_T __value, ::std::uint32_t __radix_offset)
{
    return (__value >> __radix_offset) & _T(__radix_mask);
}


template <typename _KernelName, ::std::uint32_t __radix_bits, bool __is_ascending, 
          typename valT, typename countT
          >
sycl::event
count_submit(sycl::queue exec_q, ::std::size_t __segments, ::std::size_t __block_size,
	     ::std::uint32_t __radix_offset, 
	     const valT *__val_ptr, size_t __val_size,
	     countT *__count_ptr, 
	     std::vector<sycl::event> const &__dependency_event
)
{
    // typedefs
    using _CountT = countT;

    // radix states used for an array storing bucket state counters
    constexpr ::std::uint32_t __radix_states = ::std::uint32_t(1) << __radix_bits;

    const ::std::size_t __val_buf_size = __val_size;
    // iteration space info
    const ::std::size_t __blocks_total = __ceiling_div(__val_buf_size, __block_size);
    const ::std::size_t __blocks_per_segment = __ceiling_div(__blocks_total, __segments);

    // submit to compute arrays with local count values
    sycl::event __count_levent = exec_q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // ensure the input data and the space for counters are accessible
        // oneapi::dpl::__ranges::__require_access(__hdl, __val_rng, __count_rng);
        // an accessor per work-group with value counters from each work-item
        auto __count_lacc = sycl::local_accessor<_CountT, 1>(__block_size * __radix_states, __hdl);
        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__segments * __block_size, __block_size), [=](sycl::nd_item<1> __self_item) {
                // item info
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __seg_start = __blocks_per_segment * __block_size * __wgroup_idx;

                // 1.1. count per witem: create a private array for storing count values
                _CountT __count_arr[__radix_states] = {0};
                // 1.2. count per witem: count values and write result to private count array
                const ::std::size_t __seg_end =
                    sycl::min(__seg_start + __block_size * __blocks_per_segment, __val_buf_size);
                for (::std::size_t __val_idx = __seg_start + __self_lidx; __val_idx < __seg_end;
                     __val_idx += __block_size)
                {
                    // get the bucket for the bit-ordered input value, applying the offset and mask for radix bits
                    auto __val = __order_preserving_cast<__is_ascending>(__val_ptr[__val_idx]);
                    ::std::uint32_t __bucket = __get_bucket<(1 << __radix_bits) - 1>(__val, __radix_offset);
                    // increment counter for this bit bucket
                    ++__count_arr[__bucket];
                }
                // 1.3. count per witem: write private count array to local count array
                const ::std::uint32_t __count_start_idx = __radix_states * __self_lidx;
                for (::std::uint32_t __radix_state_idx = 0; __radix_state_idx < __radix_states; ++__radix_state_idx)
                    __count_lacc[__count_start_idx + __radix_state_idx] = __count_arr[__radix_state_idx];
		__self_item.barrier(sycl::access::fence_space::local_space);

                // 2.1. count per wgroup: reduce till __count_lacc[] size > __block_size (all threads work)
                for (::std::uint32_t __i = 1; __i < __radix_states; ++__i)
                    __count_lacc[__self_lidx] += __count_lacc[__block_size * __i + __self_lidx];
		__self_item.barrier(sycl::access::fence_space::local_space);
                // 2.2. count per wgroup: reduce until __count_lacc[] size > __radix_states (threads /= 2 per iteration)
                for (::std::uint32_t __active_ths = __block_size >> 1; __active_ths >= __radix_states;
                     __active_ths >>= 1)
                {
                    if (__self_lidx < __active_ths)
                        __count_lacc[__self_lidx] += __count_lacc[__active_ths + __self_lidx];
		    __self_item.barrier(sycl::access::fence_space::local_space);
                }
                // 2.3. count per wgroup: write local count array to global count array
                if (__self_lidx < __radix_states)
                {
                    // move buckets with the same id to adjacent positions,
                    // thus splitting __count_rng into __radix_states regions
                    __count_ptr[(__segments + 1) * __self_lidx + __wgroup_idx] = __count_lacc[__self_lidx];
                }
            });
    });

    return __count_levent;
}

} // namespace radix_sort
