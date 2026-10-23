import json

from emmy.compiler.graph import Graph


with open("problems/p00_scalar_add/k_add_pointwise.ir.json") as f:
    data = json.load(f)


graph = Graph.from_dict(data)


print ("Graph nodes:" , list(graph.nodes.keys()))


kernel = graph.nodes["add"]

print("Oprtation type:", type(kernel.op).__name__)




# inspecting the kernelOp body

body = kernel.op.body

print("Body statement count:", len(body))

print("first statement type:" , type(body[0]).__name__)



#inspecting the oprtaion inside tile

tile = body[0]

print("tile body count:", len(tile.body))
print("first operation", type(tile.body[0]).__name__)

#inspecting first load
load = tile.body[0]

print("Load destination:", load.names)
print("Load source:", load.input)


#index expression
print("Load index:", load.index)
print("Index expression type:", type(load.index[0]).__name__)



def walk_expr(expr):

    kind = type(expr).__name__

    if kind == "BinaryExpr":
        walk_expr(expr.left)
        walk_expr(expr.right)
        print("Operation:", expr.op)

    elif kind == "Var":
        print("variable:", expr.name)


    elif kind == "Literal":
        print("Literal", expr.value)


    else:
        raise TypeError(f"Unsupported expression: {kind}")


walk_expr(load.index[0])




#Generating ptx


instructions = []

def compiler_expr(expr):
    kind = type(expr).__name__

    
    if kind == "Var":
        return f"%r_{expr.name}"
        
    if kind == "Literal":

        return str(expr.value)


    if kind == "BinaryExpr":

        left = compiler_expr(expr.left)

        right = compiler_expr(expr.right)

        dest = f"%r{len(instructions) + 1}"

        opcode = {
            "*": "mul.lo.s32",
            "+": "add.s32"
        }[expr.op]

        instructions.append(
            f"{opcode} {dest}, {left}, {right} ;"
        )

        return dest

    raise TypeError(f"Unsupported expression: {kind}")

# calculating global thread ID

thread_ptx = [
    ".reg .u32 %r_tid, %r_bid, %r_bdim, %r_a0;",
    "mov.u32 %r_tid, %tid.x;",
    "mov.u32 %r_bid, %ctaid.x;",
    "mov.u32 %r_bdim, %ntid.x;",
    "mad.lo.u32 %r_a0, %r_bid, %r_bdim, %r_tid;"
]
print("Thread indexing PTX:")

for instruction in thread_ptx:
    print(instruction)

result = compiler_expr(load.index[0])

for instruction in instructions:
    print(instruction)

print("Final result:", result)