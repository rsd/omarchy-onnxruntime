#!/usr/bin/env python3
"""Write the tiny ONNX model used by the ort-smoke test, using only the stdlib.

    make-model.py OUT.onnx          write the model (byte-for-byte deterministic)
    make-model.py --check FILE      decode FILE and verify its structure

The model is  Y = Add(Relu(Conv(X, W, B)), A)  with all shapes static:

    X  float32 [1, 3, 16, 16]   graph input
    W  float32 [4, 3, 3, 3]     initializer, W[o, :, :, :] = W_O[o]
    B  float32 [4]              initializer, B_O
    A  float32 [1, 4, 1, 1]     initializer, A_O (broadcast over H and W)
    Y  float32 [1, 4, 16, 16]   graph output

Conv uses a 3x3 kernel, stride 1, pads 1.  ONNX opset 17, IR version 8.
Static shapes are required by some accelerators (NPUs), and the constant
per-output-channel weights keep the expected output cheap to compute in
plain C.  W_O, B_O and A_O are mirrored in smoke.c: keep them in sync (the
hermetic test compares them).

The ONNX file is a protobuf message; the handful of message types needed are
encoded by hand below from onnx.proto (field numbers noted inline).
"""

import hashlib
import struct
import sys

# Model constants.  Multiples of 1/8 so that every value, product and partial
# sum in the network is exactly representable even in fp16.
W_O = (0.5, -0.25, 0.125, 1.0)
B_O = (0.25, -0.5, 0.0, -1.0)
A_O = (1.0, 0.5, -0.25, 2.0)

IN_SHAPE = (1, 3, 16, 16)
OUT_SHAPE = (1, 4, 16, 16)
KERNEL = (3, 3)
IR_VERSION = 8
OPSET = 17

FLOAT = 1  # TensorProto.DataType.FLOAT
ATTR_FLOAT, ATTR_INT, ATTR_INTS = 1, 2, 7  # AttributeProto.AttributeType


# ---- protobuf wire format: encoder -----------------------------------------

def varint(n):
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def f_varint(field, n):
    return varint(field << 3) + varint(n)


def f_bytes(field, data):
    return varint(field << 3 | 2) + varint(len(data)) + data


def f_str(field, s):
    return f_bytes(field, s.encode("utf-8"))


# ---- ONNX messages ----------------------------------------------------------

def tensor(name, dims, values):
    # TensorProto: dims=1, data_type=2, name=8, raw_data=9
    raw = struct.pack("<%df" % len(values), *values)
    return (b"".join(f_varint(1, d) for d in dims) + f_varint(2, FLOAT)
            + f_str(8, name) + f_bytes(9, raw))


def value_info(name, shape):
    # ValueInfoProto: name=1, type=2
    # TypeProto: tensor_type=1 / Tensor: elem_type=1, shape=2
    # TensorShapeProto: dim=1 / Dimension: dim_value=1
    dims = b"".join(f_bytes(1, f_varint(1, d)) for d in shape)
    tensor_type = f_varint(1, FLOAT) + f_bytes(2, dims)
    return f_str(1, name) + f_bytes(2, f_bytes(1, tensor_type))


def attr_ints(name, values):
    # AttributeProto: name=1, ints=8, type=20
    return (f_str(1, name) + b"".join(f_varint(8, v) for v in values)
            + f_varint(20, ATTR_INTS))


def node(op_type, name, inputs, outputs, attrs=()):
    # NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
    return (b"".join(f_str(1, i) for i in inputs)
            + b"".join(f_str(2, o) for o in outputs)
            + f_str(3, name) + f_str(4, op_type)
            + b"".join(f_bytes(5, a) for a in attrs))


def build():
    o, c, kh, kw = OUT_SHAPE[1], IN_SHAPE[1], KERNEL[0], KERNEL[1]
    weights = []
    for w in W_O:
        weights += [w] * (c * kh * kw)
    nodes = [
        node("Conv", "conv", ["X", "W", "B"], ["conv_out"], [
            attr_ints("kernel_shape", KERNEL),
            attr_ints("pads", (1, 1, 1, 1)),
            attr_ints("strides", (1, 1)),
        ]),
        node("Relu", "relu", ["conv_out"], ["relu_out"]),
        node("Add", "add", ["relu_out", "A"], ["Y"]),
    ]
    # GraphProto: node=1, name=2, initializer=5, input=11, output=12
    graph = (b"".join(f_bytes(1, n) for n in nodes)
             + f_str(2, "ort_smoke")
             + f_bytes(5, tensor("W", (o, c, kh, kw), weights))
             + f_bytes(5, tensor("B", (o,), B_O))
             + f_bytes(5, tensor("A", (1, o, 1, 1), A_O))
             + f_bytes(11, value_info("X", IN_SHAPE))
             + f_bytes(12, value_info("Y", OUT_SHAPE)))
    # ModelProto: ir_version=1, producer_name=2, graph=7, opset_import=8
    # OperatorSetIdProto: domain=1, version=2
    return (f_varint(1, IR_VERSION) + f_str(2, "ort-smoke")
            + f_bytes(7, graph)
            + f_bytes(8, f_str(1, "") + f_varint(2, OPSET)))


# ---- protobuf wire format: decoder (used by --check) -----------------------

class DecodeError(Exception):
    pass


def read_varint(buf, pos):
    shift = result = 0
    while True:
        if pos >= len(buf):
            raise DecodeError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise DecodeError("varint too long")


def parse(buf):
    """Split a message into [(field, wire_type, value)]; fails on any slack."""
    fields, pos = [], 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if field == 0:
            raise DecodeError("field number 0")
        if wire == 0:
            val, pos = read_varint(buf, pos)
        elif wire == 2:
            n, pos = read_varint(buf, pos)
            if pos + n > len(buf):
                raise DecodeError("length-delimited field overruns message")
            val, pos = buf[pos:pos + n], pos + n
        else:
            raise DecodeError("unexpected wire type %d" % wire)
        fields.append((field, wire, val))
    return fields


def get(fields, field):
    return [v for f, _, v in fields if f == field]


def one(fields, field, what):
    vals = get(fields, field)
    if len(vals) != 1:
        raise DecodeError("%s: expected exactly one field %d, found %d"
                          % (what, field, len(vals)))
    return vals[0]


def decode_value_info(buf):
    vi = parse(buf)
    name = one(vi, 1, "value_info").decode()
    tensor_type = parse(one(parse(one(vi, 2, "value_info")), 1, "type"))
    if one(tensor_type, 1, "tensor_type") != FLOAT:
        raise DecodeError("%s: element type is not float32" % name)
    shape = tuple(one(parse(d), 1, "dim") for d in get(parse(
        one(tensor_type, 2, "tensor_type")), 1))
    return name, shape


def check(buf):
    """Decode a model written by build() and verify it; returns a summary."""
    model = parse(buf)
    if one(model, 1, "model") != IR_VERSION:
        raise DecodeError("ir_version is not %d" % IR_VERSION)
    opsets = [parse(o) for o in get(model, 8)]
    if [(one(o, 1, "opset").decode(), one(o, 2, "opset")) for o in opsets] \
            != [("", OPSET)]:
        raise DecodeError("opset_import is not [('', %d)]" % OPSET)
    graph = parse(one(model, 7, "model"))

    nodes = [parse(n) for n in get(graph, 1)]
    ops = [one(n, 4, "node").decode() for n in nodes]
    if ops != ["Conv", "Relu", "Add"]:
        raise DecodeError("unexpected node list %r" % ops)
    # Every node input must be the graph input, an initializer or an earlier output.
    inits = {}
    for t in get(graph, 5):
        tp = parse(t)
        dims = tuple(get(tp, 1))
        raw = one(tp, 9, "tensor")
        n = 1
        for d in dims:
            n *= d
        if len(raw) != 4 * n or one(tp, 2, "tensor") != FLOAT:
            raise DecodeError("tensor %s: raw_data does not match dims"
                              % one(tp, 8, "tensor").decode())
        inits[one(tp, 8, "tensor").decode()] = (dims, struct.unpack(
            "<%df" % n, raw))
    inputs = [decode_value_info(v) for v in get(graph, 11)]
    outputs = [decode_value_info(v) for v in get(graph, 12)]
    if inputs != [("X", IN_SHAPE)] or outputs != [("Y", OUT_SHAPE)]:
        raise DecodeError("unexpected graph input/output: %r %r"
                          % (inputs, outputs))
    known = {"X"} | set(inits)
    for n in nodes:
        for i in get(n, 1):
            if i.decode() not in known:
                raise DecodeError("node input %r is not defined" % i.decode())
        known |= {o.decode() for o in get(n, 2)}
    if "Y" not in known:
        raise DecodeError("graph output Y is never produced")
    w_dims, w = inits["W"]
    if w_dims != (4, 3, 3, 3) or inits["B"][0] != (4,) \
            or inits["A"][0] != (1, 4, 1, 1):
        raise DecodeError("unexpected initializer shapes")
    per_o = 3 * 3 * 3
    if tuple(w[k * per_o] for k in range(4)) != W_O \
            or any(set(w[k * per_o:(k + 1) * per_o]) != {W_O[k]}
                   for k in range(4)) \
            or inits["B"][1] != B_O or inits["A"][1] != A_O:
        raise DecodeError("initializer values differ from W_O/B_O/A_O")
    return "ok: %d bytes, ir_version %d, opset %d, nodes %s, sha256 %s" % (
        len(buf), IR_VERSION, OPSET, "->".join(ops),
        hashlib.sha256(buf).hexdigest())


def main(argv):
    if len(argv) == 3 and argv[1] == "--check":
        try:
            with open(argv[2], "rb") as f:
                print(check(f.read()))
        except (OSError, DecodeError) as e:
            print("make-model.py: %s: %s" % (argv[2], e), file=sys.stderr)
            return 1
        return 0
    if len(argv) == 2 and not argv[1].startswith("-"):
        with open(argv[1], "wb") as f:
            f.write(build())
        return 0
    print("usage: make-model.py OUT.onnx | make-model.py --check FILE",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
