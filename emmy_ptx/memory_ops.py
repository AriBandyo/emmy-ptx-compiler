def emit_smem(op, shared_declarations):
    if op.dtype != "float":
        raise NotImplementedError(
            f"Unsupported shared dtype: {op.dtype}"
        )



    num_elements = 1
    for extent in op.extents:
        num_elements *= int(extent)

    size_bytes = num_elements * 4

    shared_declarations.append(
        f".shared .align 4 .b8 {op.name}[{size_bytes}];"
    )


def emit_sync(body_instructions):
    body_instructions.append("bar.sync 0;")
