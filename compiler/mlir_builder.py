import triton._C.libtriton as libtriton

def main():
    ir = libtriton.ir
    ctx = ir.context()
    builder = ir.builder(ctx)
    module = builder.create_module()
    print(module.str())

if __name__ == "__main__":
    main()
