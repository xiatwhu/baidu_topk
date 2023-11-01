#pragma once

#include <cub/cub.cuh>
#include <cuda/std/type_traits>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

namespace cub { 

namespace detail {

template <typename Invokable, typename... Args>
using invoke_result_t = typename ::cuda::std::result_of<Invokable(Args...)>::type;

template <bool Test, class T1, class T2>
using conditional_t = typename std::conditional<Test, T1, T2>::type;

template <typename Iterator>
using value_t = typename std::iterator_traits<Iterator>::value_type;

template <typename Invokable, typename InitT, typename InputT>
using accumulator_t = 
    typename ::cuda::std::decay<invoke_result_t<Invokable, InitT, InputT>>::type;

template <typename T>
using is_integral_or_enum = 
    std::integral_constant<bool,
                         std::is_integral<T>::value || std::is_enum<T>::value>;

}   // namespace detail

template <typename T, typename U>
constexpr __host__ __device__ auto min (T &&t, U &&u)
  -> decltype(t < u ? std::forward<T>(t) : std::forward<U>(u))
{
  return t < u ? std::forward<T>(t) : std::forward<U>(u);
}

template <typename T, typename U>
constexpr __host__ __device__ auto max (T &&t, U &&u)
  -> decltype(t < u ? std::forward<U>(u) : std::forward<T>(t))
{
  return t < u ? std::forward<U>(u) : std::forward<T>(t);
}

constexpr __device__ __host__ int
Nominal4BItemsToItemsCombined(int nominal_4b_items_per_thread, int combined_bytes)
{
  return (cub::min)(nominal_4b_items_per_thread,
                    (cub::max)(1,
                               nominal_4b_items_per_thread * 8 /
                               combined_bytes));
}

template <typename NumeratorT, typename DenominatorT>
__host__ __device__ __forceinline__ constexpr NumeratorT
DivideAndRoundUp(NumeratorT n, DenominatorT d)
{
  static_assert(cub::detail::is_integral_or_enum<NumeratorT>::value &&
                cub::detail::is_integral_or_enum<DenominatorT>::value,
                "DivideAndRoundUp is only intended for integral types.");

  // Static cast to undo integral promotion.
  return static_cast<NumeratorT>(n / d + (n % d != 0 ? 1 : 0));
}

template <int                      _BLOCK_THREADS,
          int                      _ITEMS_PER_THREAD = 1,
          BlockLoadAlgorithm       _LOAD_ALGORITHM   = BLOCK_LOAD_DIRECT,
          CacheLoadModifier        _LOAD_MODIFIER    = LOAD_DEFAULT,
          BlockScanAlgorithm       _SCAN_ALGORITHM   = BLOCK_SCAN_WARP_SCANS,
          BlockStoreAlgorithm      _STORE_ALGORITHM  = BLOCK_STORE_DIRECT>
struct AgentScanByKeyPolicy
{
    enum
    {
        BLOCK_THREADS    = _BLOCK_THREADS,
        ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
    };

    static const BlockLoadAlgorithm  LOAD_ALGORITHM  = _LOAD_ALGORITHM;
    static const CacheLoadModifier   LOAD_MODIFIER   = _LOAD_MODIFIER;
    static const BlockScanAlgorithm  SCAN_ALGORITHM  = _SCAN_ALGORITHM;
    static const BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
};

template <
    typename AgentScanByKeyPolicyT,       ///< Parameterized AgentScanPolicyT tuning policy type
    typename KeysInputIteratorT,          ///< Random-access input iterator type
    typename ValuesInputIteratorT,        ///< Random-access input iterator type
    typename ValuesOutputIteratorT,       ///< Random-access output iterator type
    typename EqualityOp,                  ///< Equality functor type
    typename ScanOpT,                     ///< Scan functor type
    typename InitValueT,                  ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT>                     ///< Signed integer type for global offsets
struct AgentScanByKey
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    using KeyT = cub::detail::value_t<KeysInputIteratorT>;
    using InputT = cub::detail::value_t<ValuesInputIteratorT>;

    // The output value type -- used as the intermediate accumulator
    // Per https://wg21.link/P0571, use InitValueT if provided, otherwise the
    // input iterator's value type.
    using OutputT =
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 InputT,
                                 InitValueT>;

    using SizeValuePairT = KeyValuePair<OffsetT, OutputT>;
    using KeyValuePairT = KeyValuePair<KeyT, OutputT>;
    using ReduceBySegmentOpT = ReduceBySegmentOp<ScanOpT>;

    using ScanTileStateT = ReduceByKeyScanTileState<OutputT, OffsetT>;

    // Constants
    enum
    {
        IS_INCLUSIVE        = std::is_same<InitValueT, NullType>::value, // Inclusive scan if no init_value type is provided
        BLOCK_THREADS       = AgentScanByKeyPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentScanByKeyPolicyT::ITEMS_PER_THREAD,
        ITEMS_PER_TILE      = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    using WrappedKeysInputIteratorT = cub::detail::conditional_t<std::is_pointer<KeysInputIteratorT>::value,
        CacheModifiedInputIterator<AgentScanByKeyPolicyT::LOAD_MODIFIER, KeyT, OffsetT>,   // Wrap the native input pointer with CacheModifiedInputIterator
        KeysInputIteratorT>;
    using WrappedValuesInputIteratorT = cub::detail::conditional_t<std::is_pointer<ValuesInputIteratorT>::value,
        CacheModifiedInputIterator<AgentScanByKeyPolicyT::LOAD_MODIFIER, InputT, OffsetT>,   // Wrap the native input pointer with CacheModifiedInputIterator
        ValuesInputIteratorT>;

    using BlockLoadKeysT = BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentScanByKeyPolicyT::LOAD_ALGORITHM>;
    using BlockLoadValuesT = BlockLoad<OutputT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentScanByKeyPolicyT::LOAD_ALGORITHM>;
    using BlockStoreValuesT = BlockStore<OutputT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentScanByKeyPolicyT::STORE_ALGORITHM>;
    using BlockDiscontinuityKeysT = BlockDiscontinuity<KeyT, BLOCK_THREADS, 1, 1>;

    using TilePrefixCallbackT = TilePrefixCallbackOp<SizeValuePairT, ReduceBySegmentOpT, ScanTileStateT>;
    using BlockScanT = BlockScan<SizeValuePairT, BLOCK_THREADS, AgentScanByKeyPolicyT::SCAN_ALGORITHM, 1, 1>;

    union TempStorage_
    {
        struct ScanStorage
        {
            typename BlockScanT::TempStorage              scan;
            typename TilePrefixCallbackT::TempStorage     prefix;
            typename BlockDiscontinuityKeysT::TempStorage discontinuity;
        } scan_storage;

        typename BlockLoadKeysT::TempStorage    load_keys;
        typename BlockLoadValuesT::TempStorage  load_values;
        typename BlockStoreValuesT::TempStorage store_values;
    };

    struct TempStorage : cub::Uninitialized<TempStorage_> {};

    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    TempStorage_                 &storage;
    WrappedKeysInputIteratorT     d_keys_in;
    WrappedValuesInputIteratorT   d_values_in;
    ValuesOutputIteratorT         d_values_out;
    InequalityWrapper<EqualityOp> inequality_op;
    ScanOpT                       scan_op;
    ReduceBySegmentOpT            pair_scan_op;
    InitValueT                    init_value;

    //---------------------------------------------------------------------
    // Block scan utility methods (first tile)
    //---------------------------------------------------------------------

    // Exclusive scan specialization
    __device__ __forceinline__
    void ScanTile(
        SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
        SizeValuePairT &tile_aggregate,
        Int2Type<false> /* is_inclusive */)
    {
        BlockScanT(storage.scan_storage.scan)
            .ExclusiveScan(scan_items, scan_items, pair_scan_op, tile_aggregate);
    }

    // Inclusive scan specialization
    __device__ __forceinline__
    void ScanTile(
        SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
        SizeValuePairT &tile_aggregate,
        Int2Type<true> /* is_inclusive */)
    {
        BlockScanT(storage.scan_storage.scan)
            .InclusiveScan(scan_items, scan_items, pair_scan_op, tile_aggregate);
    }

    //---------------------------------------------------------------------
    // Block scan utility methods (subsequent tiles)
    //---------------------------------------------------------------------

    // Exclusive scan specialization (with prefix from predecessors)
    __device__ __forceinline__
    void ScanTile(
        SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
        SizeValuePairT & tile_aggregate,
        TilePrefixCallbackT &prefix_op,
        Int2Type<false> /* is_incclusive */)
    {
        BlockScanT(storage.scan_storage.scan)
            .ExclusiveScan(scan_items, scan_items, pair_scan_op, prefix_op);
        tile_aggregate = prefix_op.GetBlockAggregate();
    }

    // Inclusive scan specialization (with prefix from predecessors)
    __device__ __forceinline__
    void ScanTile(
        SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
        SizeValuePairT & tile_aggregate,
        TilePrefixCallbackT &prefix_op,
        Int2Type<true> /* is_inclusive */)
    {
        BlockScanT(storage.scan_storage.scan)
            .InclusiveScan(scan_items, scan_items, pair_scan_op, prefix_op);
        tile_aggregate = prefix_op.GetBlockAggregate();
    }

    //---------------------------------------------------------------------
    // Zip utility methods
    //---------------------------------------------------------------------

    template <bool IS_LAST_TILE>
    __device__ __forceinline__
    void ZipValuesAndFlags(
        OffsetT num_remaining,
        OutputT (&values)[ITEMS_PER_THREAD],
        OffsetT (&segment_flags)[ITEMS_PER_THREAD],
        SizeValuePairT (&scan_items)[ITEMS_PER_THREAD])
    {
        // Zip values and segment_flags
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Set segment_flags for first out-of-bounds item, zero for others
            if (IS_LAST_TILE &&
                OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining)
                segment_flags[ITEM] = 1;

            scan_items[ITEM].value = values[ITEM];
            scan_items[ITEM].key   = segment_flags[ITEM];
        }
    }

    __device__ __forceinline__
    void UnzipValues(
        OutputT (&values)[ITEMS_PER_THREAD],
        SizeValuePairT (&scan_items)[ITEMS_PER_THREAD])
    {
        // Zip values and segment_flags
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            values[ITEM] = scan_items[ITEM].value;
        }
    }

    template <bool IsNull = std::is_same<InitValueT, NullType>::value,
              typename std::enable_if<!IsNull, int>::type = 0>
    __device__ __forceinline__ void AddInitToScan(
        OutputT (&items)[ITEMS_PER_THREAD],
        OffsetT (&flags)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            items[ITEM] = flags[ITEM] ? init_value : scan_op(init_value, items[ITEM]);
        }
    }

    template <bool IsNull = std::is_same<InitValueT, NullType>::value,
              typename std::enable_if<IsNull, int>::type = 0>
    __device__ __forceinline__
    void AddInitToScan(
        OutputT (&/*items*/)[ITEMS_PER_THREAD],
        OffsetT (&/*flags*/)[ITEMS_PER_THREAD])
    {}

    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    // Process a tile of input (dynamic chained scan)
    //
    template <bool IS_LAST_TILE>
    __device__ __forceinline__
    void ConsumeTile(
        OffsetT          /*num_items*/,
        OffsetT          num_remaining,
        int              tile_idx,
        OffsetT          tile_base,
        ScanTileStateT&  tile_state)
    {
        // Load items
        KeyT           keys[ITEMS_PER_THREAD];
        OutputT        values[ITEMS_PER_THREAD];
        OffsetT        segment_flags[ITEMS_PER_THREAD];
        SizeValuePairT scan_items[ITEMS_PER_THREAD];

        if (IS_LAST_TILE)
        {
            // Fill last element with the first element
            // because collectives are not suffix guarded
            BlockLoadKeysT(storage.load_keys)
                .Load(d_keys_in + tile_base,
                      keys,
                      num_remaining,
                      *(d_keys_in + tile_base));
        }
        else
        {
            BlockLoadKeysT(storage.load_keys)
                .Load(d_keys_in + tile_base, keys);
        }

        CTA_SYNC();

        if (IS_LAST_TILE)
        {
            // Fill last element with the first element
            // because collectives are not suffix guarded
            BlockLoadValuesT(storage.load_values)
                .Load(d_values_in + tile_base,
                      values,
                      num_remaining,
                      *(d_values_in + tile_base));
        }
        else
        {
            BlockLoadValuesT(storage.load_values)
                .Load(d_values_in + tile_base, values);
        }

        CTA_SYNC();

        // first tile
        if (tile_idx == 0)
        {
            BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
                .FlagHeads(segment_flags, keys, inequality_op);

            // Zip values and segment_flags
            ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                            values,
                                            segment_flags,
                                            scan_items);

            // Exclusive scan of values and segment_flags
            SizeValuePairT tile_aggregate;
            ScanTile(scan_items, tile_aggregate, Int2Type<IS_INCLUSIVE>());

            if (threadIdx.x == 0)
            {
                if (!IS_LAST_TILE)
                    tile_state.SetInclusive(0, tile_aggregate);

                scan_items[0].key = 0;
            }
        }
        else
        {
            KeyT tile_pred_key = (threadIdx.x == 0) ? d_keys_in[tile_base - 1] : KeyT();
            BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
                .FlagHeads(segment_flags, keys, inequality_op, tile_pred_key);

            // Zip values and segment_flags
            ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                            values,
                                            segment_flags,
                                            scan_items);

            SizeValuePairT  tile_aggregate;
            TilePrefixCallbackT prefix_op(tile_state, storage.scan_storage.prefix, pair_scan_op, tile_idx);
            ScanTile(scan_items, tile_aggregate, prefix_op, Int2Type<IS_INCLUSIVE>());
        }

        CTA_SYNC();

        UnzipValues(values, scan_items);

        AddInitToScan(values, segment_flags);

        // Store items
        if (IS_LAST_TILE)
        {
            BlockStoreValuesT(storage.store_values)
                .Store(d_values_out + tile_base, values, num_remaining);
        }
        else
        {
            BlockStoreValuesT(storage.store_values)
                .Store(d_values_out + tile_base, values);
        }
    }

    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Dequeue and scan tiles of items as part of a dynamic chained scan
    // with Init functor
    __device__ __forceinline__
    AgentScanByKey(
        TempStorage &         storage,
        KeysInputIteratorT    d_keys_in,
        ValuesInputIteratorT  d_values_in,
        ValuesOutputIteratorT d_values_out,
        EqualityOp            equality_op,
        ScanOpT               scan_op,
        InitValueT            init_value)
    : 
        storage(storage.Alias()),
        d_keys_in(d_keys_in),
        d_values_in(d_values_in),
        d_values_out(d_values_out),
        inequality_op(equality_op),
        scan_op(scan_op),
        pair_scan_op(scan_op),
        init_value(init_value)
    {}
    
    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        OffsetT             num_items,          ///< Total number of input items
        ScanTileStateT&     tile_state,         ///< Global tile state descriptor
        int                 start_tile)         ///< The starting tile for the current grid
    {
        int  tile_idx         = blockIdx.x;
        OffsetT tile_base     = OffsetT(ITEMS_PER_TILE) * tile_idx;
        OffsetT num_remaining = num_items - tile_base;

        if (num_remaining > ITEMS_PER_TILE)
        {
            // Not the last tile (full)
            ConsumeTile<false>(num_items,
                               num_remaining,
                               tile_idx,
                               tile_base,
                               tile_state);
        }
        else if (num_remaining > 0)
        {
            // The last tile (possibly partially-full)
            ConsumeTile<true>(num_items,
                              num_remaining,
                              tile_idx,
                              tile_base,
                              tile_state);
        }
    }
};

/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename ChainedPolicyT,              ///< Chained tuning policy
    typename KeysInputIteratorT,          ///< Random-access input iterator type
    typename ValuesInputIteratorT,        ///< Random-access input iterator type
    typename ValuesOutputIteratorT,       ///< Random-access output iterator type
    typename ScanByKeyTileStateT,         ///< Tile status interface type
    typename EqualityOp,                  ///< Equality functor type
    typename ScanOpT,                     ///< Scan functor type
    typename InitValueT,                  ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT>                     ///< Signed integer type for global offsets
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT::BLOCK_THREADS))
__global__ void DeviceScanByKeyKernel(
    KeysInputIteratorT    d_keys_in,          ///< Input keys data
    ValuesInputIteratorT  d_values_in,        ///< Input values data
    ValuesOutputIteratorT d_values_out,       ///< Output values data
    ScanByKeyTileStateT   tile_state,         ///< Tile status interface
    int                   start_tile,         ///< The starting tile for the current grid
    EqualityOp            equality_op,        ///< Binary equality functor
    ScanOpT               scan_op,            ///< Binary scan functor
    InitValueT            init_value,         ///< Initial value to seed the exclusive scan
    OffsetT               num_items)          ///< Total number of scan items for the entire problem
{
    typedef typename ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT ScanByKeyPolicyT;

    // Thread block type for scanning input tiles
    typedef AgentScanByKey<
        ScanByKeyPolicyT,
        KeysInputIteratorT,
        ValuesInputIteratorT,
        ValuesOutputIteratorT,
        EqualityOp,
        ScanOpT,
        InitValueT,
        OffsetT> AgentScanByKeyT;

    // Shared memory for AgentScanByKey
    __shared__ typename AgentScanByKeyT::TempStorage temp_storage;

    // Process tiles
    AgentScanByKeyT(
        temp_storage,
        d_keys_in,
        d_values_in,
        d_values_out,
        equality_op,
        scan_op,
        init_value
    ).ConsumeRange(
        num_items,
        tile_state,
        start_tile);
}


/******************************************************************************
 * Policy
 ******************************************************************************/

template <typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename InitValueT>
struct DeviceScanByKeyPolicy
{
    using KeyT = cub::detail::value_t<KeysInputIteratorT>;
    using ValueT = cub::detail::conditional_t<
        std::is_same<InitValueT, NullType>::value,
        cub::detail::value_t<ValuesInputIteratorT>,
        InitValueT>;
    static constexpr size_t MaxInputBytes = (sizeof(KeyT) > sizeof(ValueT) ? sizeof(KeyT) : sizeof(ValueT));
    static constexpr size_t CombinedInputBytes = sizeof(KeyT) + sizeof(ValueT);

    // SM350
    struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
    {
        enum
        {
            NOMINAL_4B_ITEMS_PER_THREAD = 6,
            ITEMS_PER_THREAD = ((MaxInputBytes <= 8) ? 6 :
                Nominal4BItemsToItemsCombined(NOMINAL_4B_ITEMS_PER_THREAD, CombinedInputBytes)),
        };

        typedef AgentScanByKeyPolicy<
                128, ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS,
                BLOCK_STORE_WARP_TRANSPOSE>
            ScanByKeyPolicyT;
    };

    // SM520
    struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
    {
        enum
        {
            NOMINAL_4B_ITEMS_PER_THREAD = 9,

            ITEMS_PER_THREAD = ((MaxInputBytes <= 8) ? 9 :
                Nominal4BItemsToItemsCombined(NOMINAL_4B_ITEMS_PER_THREAD, CombinedInputBytes)),
        };

        typedef AgentScanByKeyPolicy<
                256, ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS,
                BLOCK_STORE_WARP_TRANSPOSE>
            ScanByKeyPolicyT;
    };

    /// MaxPolicy
    typedef Policy520 MaxPolicy;
};


/******************************************************************************
 * Dispatch
 ******************************************************************************/


/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceScan
 */
template <
    typename KeysInputIteratorT,          ///< Random-access input iterator type
    typename ValuesInputIteratorT,        ///< Random-access input iterator type
    typename ValuesOutputIteratorT,       ///< Random-access output iterator type
    typename EqualityOp,                  ///< Equality functor type
    typename ScanOpT,                     ///< Scan functor type
    typename InitValueT,                  ///< The init_value element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT,                     ///< Signed integer type for global offsets
    typename SelectedPolicy = DeviceScanByKeyPolicy<KeysInputIteratorT, ValuesInputIteratorT, InitValueT>
>
struct DispatchScanByKey:
    SelectedPolicy
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128
    };

    // The input key type
    using KeyT = cub::detail::value_t<KeysInputIteratorT>;

    // The input value type
    using InputT = cub::detail::value_t<ValuesInputIteratorT>;

    // The output value type -- used as the intermediate accumulator
    // Per https://wg21.link/P0571, use InitValueT if provided, otherwise the
    // input iterator's value type.
    using OutputT =
      cub::detail::conditional_t<std::is_same<InitValueT, NullType>::value,
                                 InputT,
                                 InitValueT>;

    void*                 d_temp_storage;         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&               temp_storage_bytes;     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    KeysInputIteratorT    d_keys_in;              ///< [in] Iterator to the input sequence of key items
    ValuesInputIteratorT  d_values_in;            ///< [in] Iterator to the input sequence of value items
    ValuesOutputIteratorT d_values_out;           ///< [out] Iterator to the input sequence of value items
    EqualityOp            equality_op;            ///< [in]Binary equality functor
    ScanOpT               scan_op;                ///< [in] Binary scan functor
    InitValueT            init_value;             ///< [in] Initial value to seed the exclusive scan
    OffsetT               num_items;              ///< [in] Total number of input items (i.e., the length of \p d_in)
    cudaStream_t          stream;                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                  debug_synchronous;
    int                   ptx_version;

    CUB_RUNTIME_FUNCTION __forceinline__
    DispatchScanByKey(
        void*                 d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&               temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,              ///< [in] Iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,            ///< [in] Iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,           ///< [out] Iterator to the input sequence of value items
        EqualityOp            equality_op,            ///< [in] Binary equality functor
        ScanOpT               scan_op,                ///< [in] Binary scan functor
        InitValueT            init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT               num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t          stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous,
        int                   ptx_version
    ):
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_keys_in(d_keys_in),
        d_values_in(d_values_in),
        d_values_out(d_values_out),
        equality_op(equality_op),
        scan_op(scan_op),
        init_value(init_value),
        num_items(num_items),
        stream(stream),
        debug_synchronous(debug_synchronous),
        ptx_version(ptx_version)
    {}

    template <typename ActivePolicyT, typename InitKernel, typename ScanKernel>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke(InitKernel init_kernel, ScanKernel scan_kernel)
    {
#ifndef CUB_RUNTIME_ENABLED

        (void)init_kernel;
        (void)scan_kernel;

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else
        typedef typename ActivePolicyT::ScanByKeyPolicyT Policy;
        typedef ReduceByKeyScanTileState<OutputT, OffsetT> ScanByKeyTileStateT;

        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Number of input tiles
            int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
            int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

            // Specify temporary storage allocation requirements
            size_t  allocation_sizes[1];
            if (CubDebug(error = ScanByKeyTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors

            // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
            void* allocations[1] = {};
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                break;
            }

            // Return if empty problem
            if (num_items == 0)
                break;

            // Construct the tile status interface
            ScanByKeyTileStateT tile_state;
            if (CubDebug(error = tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;

            // Log init_kernel configuration
            int init_grid_size = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
            if (debug_synchronous) _CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors
            thrust::cuda_cub::launcher::triple_chevron(
                init_grid_size, INIT_KERNEL_THREADS, 0, stream
            ).doit(init_kernel, tile_state, num_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;


            // Get SM occupancy for scan_kernel
            int scan_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                scan_sm_occupancy,            // out
                scan_kernel,
                Policy::BLOCK_THREADS))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Run grids in epochs (in case number of tiles exceeds max x-dimension
            int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
            for (int start_tile = 0; start_tile < num_tiles; start_tile += scan_grid_size)
            {
                // Log scan_kernel configuration
                if (debug_synchronous) _CubLog("Invoking %d scan_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                    start_tile, scan_grid_size, Policy::BLOCK_THREADS, (long long) stream, Policy::ITEMS_PER_THREAD, scan_sm_occupancy);

                // Invoke scan_kernel
                thrust::cuda_cub::launcher::triple_chevron(
                    scan_grid_size, Policy::BLOCK_THREADS, 0, stream
                ).doit(
                    scan_kernel,
                    d_keys_in,
                    d_values_in,
                    d_values_out,
                    tile_state,
                    start_tile,
                    equality_op,
                    scan_op,
                    init_value,
                    num_items);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
            }
        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }

    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __host__  __forceinline__
    cudaError_t Invoke()
    {
        typedef typename DispatchScanByKey::MaxPolicy MaxPolicyT;
        typedef ReduceByKeyScanTileState<OutputT, OffsetT> ScanByKeyTileStateT;
        // Ensure kernels are instantiated.
        return Invoke<ActivePolicyT>(
            DeviceScanInitKernel<ScanByKeyTileStateT>,
            DeviceScanByKeyKernel<
                MaxPolicyT, KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT,
                ScanByKeyTileStateT, EqualityOp, ScanOpT, InitValueT, OffsetT>
        );
    }


    /**
     * Internal dispatch routine
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                 d_temp_storage,         ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&               temp_storage_bytes,     ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT    d_keys_in,              ///< [in] Iterator to the input sequence of key items
        ValuesInputIteratorT  d_values_in,            ///< [in] Iterator to the input sequence of value items
        ValuesOutputIteratorT d_values_out,           ///< [out] Iterator to the input sequence of value items
        EqualityOp            equality_op,            ///< [in] Binary equality functor
        ScanOpT               scan_op,                ///< [in] Binary scan functor
        InitValueT            init_value,             ///< [in] Initial value to seed the exclusive scan
        OffsetT               num_items,              ///< [in] Total number of input items (i.e., the length of \p d_in)
        cudaStream_t          stream,                 ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                  debug_synchronous)
    {
        typedef typename DispatchScanByKey::MaxPolicy MaxPolicyT;

        cudaError_t error;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            DispatchScanByKey dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_keys_in,
                d_values_in,
                d_values_out,
                equality_op,
                scan_op,
                init_value,
                num_items,
                stream,
                debug_synchronous,
                ptx_version
            );
            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};

template <
    typename        KeysInputIteratorT,
    typename        ValuesInputIteratorT,
    typename        ValuesOutputIteratorT,
    typename        EqualityOpT = Equality>
CUB_RUNTIME_FUNCTION
static cudaError_t InclusiveSumByKey(
    void                  *d_temp_storage,              ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t                &temp_storage_bytes,          ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    KeysInputIteratorT    d_keys_in,                    ///< [in] Random-access input iterator to the input sequence of key items
    ValuesInputIteratorT  d_values_in,                  ///< [in] Random-access input iterator to the input sequence of value items
    ValuesOutputIteratorT d_values_out,                 ///< [out] Random-access output iterator to the output sequence of value items
    int                   num_items,                    ///< [in] Total number of input items (i.e., the length of \p d_keys_in and \p d_values_in)
    EqualityOpT           equality_op = EqualityOpT(),  ///< [in] Binary functor that defines the equality of keys. Default is cub::Equality().
    cudaStream_t          stream=0,                     ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                  debug_synchronous=false)      ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
{
    // Signed integer type for global offsets
    typedef int OffsetT;

    return DispatchScanByKey<
        KeysInputIteratorT, ValuesInputIteratorT, ValuesOutputIteratorT, EqualityOpT, Sum, NullType, OffsetT>
    ::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_values_in,
        d_values_out,
        equality_op,
        Sum(),
        NullType(),
        num_items,
        stream,
        debug_synchronous);
}

}   // namespace cub