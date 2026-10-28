def _axis_only(expr, axes):
    kind = type(expr).__name__

    if kind == "Literal":
        return True

    if kind == "Var":
        return expr.name in axes

    if kind == "BinaryExpr":
        return (
            _axis_only(expr.left, axes)
            and _axis_only(expr.right, axes)
        )

    if kind == "TernaryExpr":
        return (
            _axis_only(expr.cond, axes)
            and _axis_only(expr.if_true, axes)
            and _axis_only(expr.if_false, axes)
        )

    return False


def schedule_tile_operations(body, axes):
    items = list(enumerate(body))
    axis_names = {axis.name for axis in axes}
    scheduled = []
    i = 0

    while i < len(items):
        kind = type(items[i][1]).__name__

        if kind not in ("Load", "Assign"):
            scheduled.append(items[i])
            i += 1
            continue

        j = i

        while j < len(items):
            kind = type(items[j][1]).__name__

            if kind not in ("Load", "Assign"):
                break

            j += 1

        group = items[i:j]

        loads = [
            item for item in group
            if type(item[1]).__name__ == "Load"
        ]

        assigns = [
            item for item in group
            if type(item[1]).__name__ == "Assign"
        ]

        safe = all(
            all(_axis_only(index, axis_names) for index in op.index)
            for _, op in loads
        )

        if safe and loads and assigns:
            scheduled.extend(loads)
            scheduled.extend(assigns)
        else:
            scheduled.extend(group)

        i = j

    return scheduled
