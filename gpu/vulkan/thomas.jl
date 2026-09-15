"""Small direct Vulkan smoke test for the frozen Thomas ABI.

This is deliberately a correctness probe, not a benchmark.  It allocates host-visible
storage buffers, submits one compute dispatch, waits for completion, and compares the
result with the scalar Julia recurrence.  The default batch is tiny so this file is safe
to run on a shared workstation:

    julia --project=gpu/vulkan gpu/vulkan/thomas.jl

The first prototype assumes queue family 0 and memory type 0, as in the Vulkan.jl
minimal-compute tutorial.  A production context must discover both properties and should
use device-local buffers plus a staging arena.
"""
using Interleave
using Vulkan

include(joinpath(@__DIR__, "abi.jl"))
include(joinpath(@__DIR__, "compile.jl"))

function thomas!(X, D, U, L, B, S)
    s = D[1]
    sm1 = inv(s)
    X[1] = B[1] * sm1
    for i in 2:length(X)
        S[i] = U[i - 1] * sm1
        s = D[i] - L[i] * S[i]
        X[i] = B[i] - L[i] * X[i - 1]
        sm1 = inv(s)
        X[i] *= sm1
    end
    for i in length(X)-1:-1:1
        X[i] -= S[i + 1] * X[i + 1]
    end
    X
end

struct VulkanThomasBuffers
    buffers::NTuple{6,Any}
    memories::NTuple{6,Any}
    mapped::NTuple{6,Any}
end

function _storage_buffer(device, queue_family, nbytes, memory_type)
    memory = DeviceMemory(device, nbytes, memory_type)
    buffer = Buffer(device, nbytes, BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    SHARING_MODE_EXCLUSIVE, [queue_family])
    unwrap(bind_buffer_memory(device, buffer, memory, 0))
    ptr = convert(Ptr{Float32}, unwrap(map_memory(device, memory, 0, nbytes)))
    host = unsafe_wrap(Vector{Float32}, ptr, nbytes ÷ sizeof(Float32); own = false)
    buffer, memory, host
end

function _write_descriptor_set(device, dset, binding, buffer)
    WriteDescriptorSet(dset, binding, 0, DESCRIPTOR_TYPE_STORAGE_BUFFER, [],
                       [DescriptorBufferInfo(buffer, 0, WHOLE_SIZE)], [])
end

function run_vulkan_thomas(; nbatch = 8, nx = 8, workgroup = 8,
                            queue_family = 0, memory_type = 0)
    nbatch > 0 || throw(ArgumentError("nbatch must be positive"))
    nx > 0 || throw(ArgumentError("nx must be positive"))
    instance = Instance([], [])
    physical = first(unwrap(enumerate_physical_devices(instance)))
    device = Device(physical, [DeviceQueueCreateInfo(queue_family, [1.0])], [], [])
    queue = get_device_queue(device, queue_family, 0)

    n = nbatch * nx
    bytes = n * sizeof(Float32)
    storage = ntuple(_ -> _storage_buffer(device, queue_family, bytes, memory_type), 6)
    buffers = ntuple(i -> storage[i][1], 6)
    memories = ntuple(i -> storage[i][2], 6)
    mapped = ntuple(i -> storage[i][3], 6)

    X = zeros(Float32, nbatch, nx)
    D = fill(2f0, nbatch, nx)
    U = fill(-1f0, nbatch, nx)
    L = fill(-1f0, nbatch, nx)
    B = [sinpi(Float32(b) / 8) + Float32(i) / nx for b in 1:nbatch, i in 1:nx]
    S = zeros(Float32, nbatch, nx)
    reference = similar(X)
    for b in 1:nbatch
        thomas!(view(reference, b, :), view(D, b, :), view(U, b, :),
                view(L, b, :), view(B, b, :), view(S, b, :))
    end

    copyto!(mapped[1], vec(X)); copyto!(mapped[2], vec(D)); copyto!(mapped[3], vec(U))
    copyto!(mapped[4], vec(L)); copyto!(mapped[5], vec(B)); fill!(mapped[6], 0f0)
    unwrap(flush_mapped_memory_ranges(device,
        [MappedMemoryRange(memories[i], 0, bytes) for i in 1:6]))

    words = reinterpret(UInt32, read(joinpath(@__DIR__, "shaders", "thomas.spv")))
    shader = ShaderModule(device, sizeof(UInt32) * length(words), words)
    bindings = [DescriptorSetLayoutBinding(i - 1, DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            SHADER_STAGE_COMPUTE_BIT;
                                            descriptor_count = 1) for i in 1:6]
    dsl = DescriptorSetLayout(device, bindings)
    pl = PipelineLayout(device, [dsl],
                        [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(ThomasPushConstants))])
    spec = [UInt32(workgroup)]
    stage = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, shader, "main",
        specialization_info = SpecializationInfo(
            [SpecializationMapEntry(0, 0, sizeof(UInt32))],
            sizeof(UInt32), Ptr{Nothing}(pointer(spec))))
    pipelines, _ = unwrap(create_compute_pipelines(device,
        [ComputePipelineCreateInfo(stage, pl, -1)]))
    pipeline = first(pipelines)
    pool = DescriptorPool(device, 1,
        [DescriptorPoolSize(DESCRIPTOR_TYPE_STORAGE_BUFFER, 6)];
        flags = DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
    dsets = unwrap(allocate_descriptor_sets(device, DescriptorSetAllocateInfo(pool, [dsl])))
    dset = first(dsets)
    update_descriptor_sets(device,
        [_write_descriptor_set(device, dset, i - 1, buffers[i]) for i in 1:6], [])

    cmdpool = CommandPool(device, queue_family)
    cbufs = unwrap(allocate_command_buffers(device,
        CommandBufferAllocateInfo(cmdpool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)))
    cbuf = first(cbufs)
    begin_command_buffer(cbuf, CommandBufferBeginInfo(
        flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    cmd_bind_pipeline(cbuf, PIPELINE_BIND_POINT_COMPUTE, pipeline)
    push = [ThomasPushConstants(nbatch, nx, nbatch)]
    cmd_push_constants(cbuf, pl, SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(ThomasPushConstants), Ptr{Nothing}(pointer(push)))
    cmd_bind_descriptor_sets(cbuf, PIPELINE_BIND_POINT_COMPUTE, pl, 0, [dset], [])
    cmd_dispatch(cbuf, cld(nbatch, workgroup), 1, 1)
    end_command_buffer(cbuf)
    GC.@preserve buffers memories mapped words spec push begin
        unwrap(queue_submit(queue, [SubmitInfo([], [], [cbuf], [])]))
        unwrap(queue_wait_idle(queue))
    end
    unwrap(invalidate_mapped_memory_ranges(device,
        [MappedMemoryRange(memories[1], 0, bytes)]))
    result = reshape(copy(mapped[1]), nbatch, nx)
    @assert result == reference
    println("Vulkan Thomas smoke test passed: ", nbatch, " × ", nx,
            " (workgroup ", workgroup, ")")
    free_command_buffers(device, cmdpool, cbufs)
    free_descriptor_sets(device, pool, dsets)
    result
end

if abspath(PROGRAM_FILE) == @__FILE__
    compile_shader()
    run_vulkan_thomas()
end
