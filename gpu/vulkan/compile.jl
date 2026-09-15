using glslang_jll: glslangValidator
using SPIRV_Tools_jll: spirv_val

"""Compile and validate the Thomas compute shader. Return the generated `.spv` path."""
function compile_shader(; input = joinpath(@__DIR__, "shaders", "thomas.comp"),
                        output = joinpath(@__DIR__, "shaders", "thomas.spv"))
    glslang = glslangValidator(identity)
    validator = spirv_val(identity)
    run(`$glslang -V --target-env vulkan1.2 -S comp -o $output $input`)
    run(`$validator --target-env vulkan1.2 $output`)
    output
end

abspath(PROGRAM_FILE) == (@__FILE__) && compile_shader()
