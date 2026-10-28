from dataclasses import dataclass


@dataclass(frozen=True)
class VectorLoad:
    names: tuple
    input: str
    index: tuple
    dtype: object


def _affine(expr, axis_name):
    """Recognize an integer expression: coefficient * axis + offset."""
    kind = type(expr).__name__

    if kind == "Literal" and isinstance(expr.value, int):
        return (0, expr.value)

    if kind == "Var" and expr.name == axis_name:
        return (1, 0)

    if kind != "BinaryExpr":
        return None

    left = _affine(expr.left, axis_name)
    right = _affine(expr.right, axis_name)

    if left is None or right is None:
        return None

    a, b = left
    c, d = right

    if expr.op == "+":
        return (a + c, b + d)

    if expr.op == "-":
        return (a - c, b - d)

    if expr.op == "*":
        if a and c:
            return None
        return (a * d + c * b, b * d)

    return None


def vectorize_tile_operations(items, axes, kernel_meta):
    """Group provably aligned, consecutive f32 loads in 1D tiles.

    Unknown patterns remain unchanged.
    """
    items = list(items)

    if len(axes) != 1:
        return items

    axis = axes[0]
    extent = int(axis.extent.as_static())

    if extent <= 0:
        return items

    shapes = {
        arg["name"]: arg["shape"]
        for arg in kernel_meta["args"]
        if arg["dtype"] == "f32"
    }

    result = []
    pos = 0

    while pos < len(items):
        kind = type(items[pos][1]).__name__

        if kind not in ("Load", "Assign"):
            result.append(items[pos])
            pos += 1
            continue

        end = pos

        while end < len(items):
            kind = type(items[end][1]).__name__
            if kind not in ("Load", "Assign"):
                break
            end += 1

        region = items[pos:end]

        # We may hoist loads past pure Assign statements only
        # when all their addresses depend on the tile axis.
        safe = all(
            (
                type(op).__name__ != "Load"
                or (
                    len(op.index) == 1
                    and _affine(op.index[0], axis.name) is not None
                )
            )
            and not (
                type(op).__name__ == "Assign"
                and op.name == axis.name
            )
            for _, op in region
        )

        if not safe:
            result.extend(region)
            pos = end
            continue

        candidates = {}

        for original_i, op in region:
            if type(op).__name__ != "Load":
                continue

            if len(op.names) != 1 or op.dtype.name != "f32":
                continue

            shape = shapes.get(op.input)

            if shape is None or len(shape) != 1:
                continue

            pair = _affine(op.index[0], axis.name)

            if pair is None:
                continue

            coefficient, offset = pair

            if coefficient < 0 or offset < 0:
                continue

            key = (op.input, coefficient)

            candidates.setdefault(key, {}).setdefault(
                offset, []
            ).append((original_i, op))

        groups = {}
        consumed = set()

        for (buffer, coefficient), offsets in candidates.items():
            # 4 f32 values = 16 bytes.
            # The starting element must be divisible by four.
            if coefficient % 4 != 0:
                continue

            for base in sorted(offsets):
                if base % 4 != 0:
                    continue

                if any(
                    len(offsets.get(base + k, [])) != 1
                    for k in range(4)
                ):
                    continue

                max_index = coefficient * (extent - 1) + base + 3

                if max_index >= shapes[buffer][0]:
                    continue

                if max_index >= 2**32:
                    continue

                members = [
                    offsets[base + k][0]
                    for k in range(4)
                ]

                if any(i in consumed for i, _ in members):
                    continue

                if len({
                    op.names[0] for _, op in members
                }) != 4:
                    continue

                first_id = min(i for i, _ in members)
                first_op = members[0][1]

                groups[first_id] = VectorLoad(
                    names=tuple(
                        op.names[0] for _, op in members
                    ),
                    input=buffer,
                    index=first_op.index,
                    dtype=first_op.dtype,
                )

                consumed.update(i for i, _ in members)

        if groups:
            # Keep loads ahead of their dependent arithmetic.
            # No stores or synchronization are crossed.
            for original_i, op in region:
                if type(op).__name__ != "Load":
                    continue

                if original_i in groups:
                    result.append(
                        (original_i, groups[original_i])
                    )
                elif original_i not in consumed:
                    result.append((original_i, op))

            result.extend(
                (i, op)
                for i, op in region
                if type(op).__name__ == "Assign"
            )
        else:
            result.extend(region)

        pos = end

    return result
