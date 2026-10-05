module {
  func.func @linear_kernel(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %0 = arith.constant 4 : i32
    %1 = tensor.empty() : tensor<4xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1 : tensor<4xi32>) {
      ^bb0(%arg5: i32):
      %3 = linalg.index 0 : index
      %4 = arith.index_cast %3 : index to i32
      linalg.yield %4 : i32
    } -> tensor<4xi32>
    %5 = arith.constant 0 : i32
    %6 = tensor.empty() : tensor<4xi32>
    %7 = linalg.fill ins(%5 : i32) outs(%6 : tensor<4xi32>) -> tensor<4xi32>
    %8 = tensor.empty() : tensor<4xi32>
    %9 = linalg.add ins(%7, %2 : tensor<4xi32>, tensor<4xi32>) outs(%8 : tensor<4xi32>) -> tensor<4xi32>
    %10 = tensor.empty() : tensor<4xf32>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9 : tensor<4xi32>) outs(%10 : tensor<4xf32>) {
      ^bb0(%arg6: i32, %arg7: f32):
      %12 = arith.index_cast %arg6 : i32 to index
      %13 = memref.load %arg1[%12] : memref<?xf32>
      linalg.yield %13 : f32
    } -> tensor<4xf32>
    %14 = arith.muli %arg0, %0 : i32
    %15 = tensor.empty() : tensor<4xi32>
    %16 = linalg.fill ins(%14 : i32) outs(%15 : tensor<4xi32>) -> tensor<4xi32>
    %17 = tensor.empty() : tensor<4xi32>
    %18 = linalg.add ins(%16, %2 : tensor<4xi32>, tensor<4xi32>) outs(%17 : tensor<4xi32>) -> tensor<4xi32>
    %19 = tensor.empty() : tensor<4xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18 : tensor<4xi32>) outs(%19 : tensor<4xf32>) {
      ^bb0(%arg8: i32, %arg9: f32):
      %21 = arith.index_cast %arg8 : i32 to index
      %22 = memref.load %arg2[%21] : memref<?xf32>
      linalg.yield %22 : f32
    } -> tensor<4xf32>
    %23 = arith.index_cast %arg0 : i32 to index
    %24 = memref.load %arg3[%23] : memref<?xf32>
    %25 = tensor.empty() : tensor<4xf32>
    %26 = linalg.mul ins(%11, %20 : tensor<4xf32>, tensor<4xf32>) outs(%25 : tensor<4xf32>) -> tensor<4xf32>
    %27 = arith.constant 0.000000e+00 : f32
    %28 = tensor.empty() : tensor<f32>
    %29 = linalg.fill ins(%27 : f32) outs(%28 : tensor<f32>) -> tensor<f32>
    %30 = linalg.reduce ins(%26 : tensor<4xf32>) outs(%29 : tensor<f32>) dimensions = [0] (%arg10: f32, %arg11: f32) {
      %31 = arith.addf %arg10, %arg11 : f32
      linalg.yield %31 : f32
    }
    %32 = tensor.extract %30[] : tensor<f32>
    %33 = arith.addf %32, %24 : f32
    %34 = arith.index_cast %arg0 : i32 to index
    memref.store %33, %arg4[%34] : memref<?xf32>
    return
  }
}
