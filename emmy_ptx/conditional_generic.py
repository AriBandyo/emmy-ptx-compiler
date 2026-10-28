from emmy_ptx.basic_ops import emit_basic_op


# Branch when the original comparison is false.
FALSE_COMPARISON = {
    "==": "ne",
    "!=": "eq",
    "<": "ge",
    "<=": "gt",
    ">": "le",
    ">=": "lt",
}


def emit_generic_cond(
    op, i, *,
    tile, value_regs, instructions,
    body_instructions, compiler_expr,
    arg_order, kernel_meta,
):
    cond = op.cond

    if (
        type(cond).__name__ != "BinaryExpr"
        or cond.op not in FALSE_COMPARISON
    ):
        raise NotImplementedError(
            f"Unsupported condition: {cond}"
        )

    allowed_vars = {"lane", "warp"}
    allowed_vars.update(axis.name for axis in tile.axes)

    def operand(expr):
        kind = type(expr).__name__

        if kind == "Var":
            if expr.name not in allowed_vars:
                raise NotImplementedError(
                    f"Unsupported predicate variable: {expr.name}"
                )
            return compiler_expr(expr)

        if kind == "Literal":
            if not isinstance(expr.value, int) or expr.value < 0:
                raise NotImplementedError(
                    "Predicate literals must be nonnegative integers"
                )
            return compiler_expr(expr)

        raise NotImplementedError(
            f"Unsupported predicate operand: {kind}"
        )

    left = operand(cond.left)
    right = operand(cond.right)
    inverse = FALSE_COMPARISON[cond.op]

    else_body = tuple(op.else_body or ())
    has_else = bool(else_body)

    skip_label = (
        f"COND_ELSE_{i}" if has_else
        else f"COND_END_{i}"
    )

    body_instructions.extend([
        f"setp.{inverse}.u32 %p_not_output, {left}, {right};",
        f"@%p_not_output bra {skip_label};",
    ])

    def emit_statements(statements):
        for inner in statements:
            kind = type(inner).__name__

            if kind == "Write" and inner.output not in arg_order:
                # Shared-memory reduction write.
                if (
                    inner.atomic
                    or len(inner.values) != 1
                    or len(inner.index) != 1
                    or getattr(inner.index[0], "name", None) != "warp"
                    or inner.value_dtype.name != "f32"
                ):
                    raise NotImplementedError(
                        "Unsupported conditional shared-memory write"
                    )

                source = value_regs[inner.values[0]]

                body_instructions.extend([
                    f"mov.u32 %r_smem_base, {inner.output};",
                    "mul.lo.u32 %r_smem_off, %r_warp, 4;",
                    "add.u32 %r_smem_addr, "
                    "%r_smem_base, %r_smem_off;",
                    f"st.shared.f32 [%r_smem_addr], {source};",
                ])

            elif kind == "Write" and not inner.index:
                # Scalar output stored directly at its base address.
                if (
                    inner.atomic
                    or len(inner.values) != 1
                    or inner.value_dtype.name != "f32"
                ):
                    raise NotImplementedError(
                        "Unsupported scalar conditional write"
                    )

                output_index = arg_order.index(inner.output)
                source = value_regs[inner.values[0]]

                body_instructions.append(
                    f"st.global.f32 [%rd_base{output_index}], {source};"
                )

            elif kind in ("Load", "Assign", "Write"):
                emit_basic_op(
                    inner, i,
                    value_regs=value_regs,
                    instructions=instructions,
                    body_instructions=body_instructions,
                    compiler_expr=compiler_expr,
                    kernel_meta=kernel_meta,
                    arg_order=arg_order,
                    ptx_float_opcodes={},
                    arithmetic_instructions=[],
                )

            else:
                raise NotImplementedError(
                    f"Unsupported conditional statement: {kind}"
                )

    emit_statements(op.body)

    if has_else:
        body_instructions.extend([
            f"bra COND_END_{i};",
            f"COND_ELSE_{i}:",
        ])
        emit_statements(else_body)

    body_instructions.append(f"COND_END_{i}:")
