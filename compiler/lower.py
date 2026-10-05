import re

from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

from kernels.mlp_kernel import linear_kernel


signature = {
    "x_ptr": "*fp32",
    "weight_ptr": "*fp32",
    "bias_ptr": "*fp32",
    "output_ptr": "*fp32",
}

constexprs = {
    "INPUT_SIZE": 4,
    "OUTPUT_SIZE": 1,
}


def compile_ttir():
    source = ASTSource(
        linear_kernel,
        signature,
        constexprs=constexprs,
    )

    target = GPUTarget("cuda", 80, 32)

    options = {
        "num_warps": 1,
        "num_ctas": 1,
    }

    from triton.compiler import compile as triton_compile

    return triton_compile(
        source,
        target=target,
        options=options,
    )


OP_MAPPING = {
    "tt.get_program_id": "linalg.program_id",
    "tt.make_range": "linalg.index_range",
    "tt.splat": "linalg.broadcast",
    "tt.addptr": "linalg.pointer_add",
    "tt.load": "linalg.load",
    "tt.store": "linalg.store",
    "tt.reshape": "linalg.reshape",
    "tt.reduce": "linalg.reduce",
    "arith.constant": "linalg.constant",
    "arith.muli": "linalg.mul",
    "arith.mulf": "linalg.mul",
    "arith.addf": "linalg.add",
}


def get_operation_name(line):
    # Handles both:
    #   tt.load
    #   "tt.reduce"
    #   arith.addf
    #   "tt.reduce.return"

    match = re.search(r'"([^"]+)"', line)
    if match:
        return match.group(1)

    for name in [
        "tt.get_program_id",
        "tt.make_range",
        "tt.splat",
        "tt.addptr",
        "tt.load",
        "tt.store",
        "tt.reshape",
        "tt.reduce.return",
        "tt.reduce",
        "tt.return",
        "arith.constant",
        "arith.muli",
        "arith.mulf",
        "arith.addf",
    ]:
        if re.search(rf"\b{re.escape(name)}\b", line):
            return name

    return None


def get_result(line):
    match = re.match(r"\s*(%\w+)\s*=", line)
    return match.group(1) if match else None


def lower_operation(line):
    name = get_operation_name(line)

    if name is None:
        return None

    # These are structural terminators, not Linalg operations.
    if name == "tt.reduce.return":
        return "    linalg.reduce_yield"

    if name == "tt.return":
        return "linalg.return"

    lowered = OP_MAPPING.get(name)

    if lowered is None:
        return f"# UNHANDLED: {name}"

    result = get_result(line)

    if result:
        return f"{result} = {lowered}"

    return lowered


def main():
    result = compile_ttir()
    ttir = result.asm["ttir"]

    lowered_ops = []

    print("=== TTIR → LINALG ===")
    print()

    inside_func = False

    for line in ttir.splitlines():
        stripped = line.strip()

        if stripped.startswith("tt.func"):
            inside_func = True
            continue

        if not inside_func:
            continue

        if stripped.startswith("#"):
            continue

        if stripped.startswith("loc("):
            continue

        lowered = lower_operation(line)

        if lowered:
            lowered_ops.append(lowered)
            print(lowered)

    with open("linalg.mlir", "w") as f:
        f.write("\n".join(lowered_ops) + "\n")

    print()
    print("Wrote linalg.mlir")


if __name__ == "__main__":
    main()
