def make_compiler_expr(instructions):
    def compiler_expr(expr):
        kind = type(expr).__name__

    
        if kind == "Var":
            return f"%r_{expr.name}"
        
        if kind == "Literal":

            return str(expr.value)

        if kind == "TernaryExpr":
            cond = expr.cond

            if type(cond).__name__ != "BinaryExpr" or cond.op != "<":
                raise NotImplementedError(
                    f"Unsupported ternary condition: {cond}"
                )

            true_reg = compiler_expr(expr.if_true)
            false_reg = compiler_expr(expr.if_false)

            left_reg = compiler_expr(cond.left)
            right_reg = compiler_expr(cond.right)

            dest = f"%r{len(instructions) + 1}"

            instructions.extend([
                f"setp.lt.u32 %p_ternary, {left_reg}, {right_reg};",
                f"selp.u32 {dest}, {true_reg}, {false_reg}, %p_ternary;"
            ])

            return dest
    
        if kind == "BinaryExpr":

            left = compiler_expr(expr.left)

            right = compiler_expr(expr.right)

            dest = f"%r{len(instructions) + 1}"

            opcode = {
                "*": "mul.lo.u32",
                "+": "add.u32",
                "-": "sub.u32",
                "/": "div.u32",
                "%": "rem.u32"
            }[expr.op]

            instructions.append(
                f"{opcode} {dest}, {left}, {right} ;"
            )

            return dest

        raise TypeError(f"Unsupported expression: {kind}")

    return compiler_expr


def buffer_shape(kernel_meta, name):
    """Find a tensor's shape using kernel metadata."""
    matches = [
        arg["shape"]
        for arg in kernel_meta["args"]
        if arg["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve shape for {name}")
    return matches[0]


def flat_index(index_exprs, shape, compiler_expr, instructions):
    """Convert row-major multidimensional indices into one offset."""
    if not index_exprs:
        if any(int(dim) != 1 for dim in shape):
            raise ValueError("Missing indices for non-scalar tensor")
        return "0"

    # A single index can already represent a flat offset.
    if len(index_exprs) == 1:
        return compiler_expr(index_exprs[0])

    if len(index_exprs) != len(shape):
        raise ValueError(
            f"Index rank {len(index_exprs)} does not match shape {shape}"
        )

    acc = compiler_expr(index_exprs[0])

    for dimension in range(1, len(index_exprs)):
        idx = compiler_expr(index_exprs[dimension])
        dest = f"%r{len(instructions) + 1}"

        instructions.append(
            f"mad.lo.u32 {dest}, {acc}, {shape[dimension]}, {idx};"
        )
        acc = dest

    return acc
