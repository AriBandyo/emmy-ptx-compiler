ZERO = "0f00000000"
ONE = "0f3F800000"
TWO = "0f40000000"
LOG2E = "0f3FB8AA3B"
NEG_LOG2E = "0fBFB8AA3B"
TWO_LOG2E = "0f4038AA3B"


BINARY = {
    "add": "add.f32",
    "subtract": "sub.f32",
    "sub": "sub.f32",
    "multiply": "mul.f32",
    "divide": "div.rn.f32",
    "maximum": "max.f32",
    "minimum": "min.f32",
}

UNARY = {
    "negative": "neg.f32",
    "abs": "abs.f32",
    "sqrt": "sqrt.rn.f32",
    "reciprocal": "rcp.rn.f32",
}


def emit_elementwise(op_name, dest, args):
    if op_name == "copy":
        return [f"mov.f32 {dest}, {args[0]};"]

    if op_name in BINARY:
        return [
            f"{BINARY[op_name]} {dest}, {args[0]}, {args[1]};"
        ]

    if op_name in UNARY:
        return [
            f"{UNARY[op_name]} {dest}, {args[0]};"
        ]

    if op_name == "relu":
        return [
            f"max.f32 {dest}, {args[0]}, {ZERO};"
        ]

    if op_name == "exp":
        return [
            f"mul.f32 {dest}, {args[0]}, {LOG2E};",
            f"ex2.approx.f32 {dest}, {dest};",
        ]

    if op_name == "sigmoid":
        return [
            f"mul.f32 {dest}, {args[0]}, {NEG_LOG2E};",
            f"ex2.approx.f32 {dest}, {dest};",
            f"add.f32 {dest}, {dest}, {ONE};",
            f"rcp.rn.f32 {dest}, {dest};",
        ]

    if op_name == "tanh":
        return [
            f"mul.f32 {dest}, {args[0]}, {TWO_LOG2E};",
            f"ex2.approx.f32 {dest}, {dest};",
            f"add.f32 {dest}, {dest}, {ONE};",
            f"div.rn.f32 {dest}, {TWO}, {dest};",
            f"sub.f32 {dest}, {ONE}, {dest};",
        ]

    raise NotImplementedError(
        f"Unsupported elementwise op: {op_name}"
    )


def emit_assign(stmt, value_regs, fresh=True):
    args = [value_regs[a] for a in stmt.args]

    if fresh or stmt.name not in value_regs:
        value_regs[stmt.name] = f"%f{len(value_regs)}"

    return emit_elementwise(
        stmt.op.name,
        value_regs[stmt.name],
        args,
    )
