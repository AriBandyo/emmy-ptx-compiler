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
