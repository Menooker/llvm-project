// RUN: mlir-opt -allow-unregistered-dialect -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --merge-alloc %s | FileCheck %s

func.func @mlp(%x: tensor<128x128xf32>, %y: tensor<128x128xf32>) -> tensor<128x128xf32> {
   // CHECK-DAG:  %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<131072xi8>
   // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
   // CHECK-DAG:  %[[VIEW_A:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<131072xi8> to memref<128x128xf32>
   %a0 = tensor.empty() : tensor<128x128xf32>
   // CHECK:      linalg.matmul ins
   // CHECK-SAME: outs(%[[VIEW_A]] : memref<128x128xf32>)
   %a = linalg.matmul ins(%x, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%a0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK-DAG:  %[[C65536:.*]] = arith.constant 65536 : index
   // CHECK-DAG:  %[[VIEW_B:.*]] = memref.view %[[ALLOC]][%[[C65536]]][] : memref<131072xi8> to memref<128x128xf32>
   %b0 = tensor.empty() : tensor<128x128xf32>
   // CHECK:      linalg.matmul ins(%[[VIEW_A]],
   // CHECK-SAME: outs(%[[VIEW_B]] : memref<128x128xf32>)
   %b = linalg.matmul ins(%a, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%b0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK-DAG:  %[[C0_2:.*]] = arith.constant 0 : index
   // CHECK-DAG:  %[[VIEW_C:.*]] = memref.view %[[ALLOC]][%[[C0_2]]][] : memref<131072xi8> to memref<128x128xf32>
   %c0 = tensor.empty() : tensor<128x128xf32>
   // CHECK:      linalg.matmul ins(%[[VIEW_B]],
   // CHECK-SAME: outs(%[[VIEW_C]] : memref<128x128xf32>)
   %c = linalg.matmul ins(%b, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%c0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK-DAG:  %[[D:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
   // CHECK:      linalg.matmul ins(%[[VIEW_C]],
   // CHECK-SAME: outs(%[[D]] : memref<128x128xf32>)
   %d0 = tensor.empty() : tensor<128x128xf32>
   %d = linalg.matmul ins(%c, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%d0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK:      return %[[D]]
   return %d : tensor<128x128xf32>
}

// CHECK-LABEL: @basic
func.func @basic() -> memref<8x64xf32> {
  // CHECK-DAG: %[[BASE:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4096xi8>
  // b is used in return, complex lifetime
  // CHECK-DAG: %[[B:.*]] = memref.alloc()
  %b = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[B]])
  "test.source"(%b)  : (memref<8x64xf32>) -> ()
  // c and d has overlapping lifetime
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C:.*]] = memref.view %[[BASE]][%[[C0]]][] : memref<4096xi8> to memref<8x64xf32>
  %c = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[C]])
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[D:.*]] = memref.view %[[BASE]][%[[C2048]]][] : memref<4096xi8> to memref<8x64xf32>
  %d = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[D]])
  "test.source"(%d)  : (memref<8x64xf32>) -> ()
  // CHECK:     "test.source"(%[[C]])
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // e can reuse the above memory
  // CHECK-DAG: %[[C0_2:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[E:.*]] = memref.view %[[BASE]][%[[C0_2]]][] : memref<4096xi8> to memref<8x64xf32>
  %e = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[E]])
  "test.source"(%e)  : (memref<8x64xf32>) -> ()
  // CHECK:     return %[[B]]
  return %b : memref<8x64xf32>
}

// CHECK-LABEL: @withloop
func.func @withloop() {
  // CHECK-DAG: %[[BASE2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6144xi8>
  // CHECK-DAG: %[[C033:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[F:.*]] = memref.view %[[BASE2]][%[[C033]]][] : memref<6144xi8> to memref<8x64xf32>
  %f = memref.alloc() : memref<8x64xf32>
  // CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[G:.*]] = memref.view %[[BASE2]][%[[C2048]]][] : memref<6144xi8> to memref<8x64xf32>
  %g = memref.alloc() : memref<8x64xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c5 step %c1 {
      // CHECK:     "test.source"(%[[F]])
      "test.source"(%f)  : (memref<8x64xf32>) -> ()
      // CHECK:     "test.source"(%[[G]])
      "test.source"(%g)  : (memref<8x64xf32>) -> ()
      // CHECK-DAG: %[[C4096:.*]] = arith.constant 4096 : index
      // CHECK-DAG: %[[H:.*]] = memref.view %[[BASE2]][%[[C4096]]][] : memref<6144xi8> to memref<8x64xf32>
      %h = memref.alloc() : memref<8x64xf32>
      // CHECK:     "test.source"(%[[H]])
      "test.source"(%h)  : (memref<8x64xf32>) -> ()
      // CHECK-DAG: %[[C4096_3:.*]] = arith.constant 4096 : index
      // CHECK-DAG: %[[J:.*]] = memref.view %[[BASE2]][%[[C4096_3]]][] : memref<6144xi8> to memref<8x64xf32>
      %j = memref.alloc() : memref<8x64xf32>
      // CHECK:     "test.source"(%[[J]])
      "test.source"(%j)  : (memref<8x64xf32>) -> ()
  }
  return
}

func.func @nested_forall(%arg0: memref<2xf32>) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %alloc = memref.alloc() : memref<2xf32>
    memref.copy %arg0, %alloc : memref<2xf32> to memref<2xf32>
    scf.forall (%i) in (%c16) {
      %alloc_0 = memref.alloc() : memref<4xf32>
      %alloc_1 = memref.alloc() : memref<8xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc) : (memref<2xf32>) -> ()
        "test.source"(%alloc_1) : (memref<8xf32>) -> ()
        %alloc_2 = memref.alloc() : memref<16xf32>
        "test.source"(%alloc_2) : (memref<16xf32>) -> ()
      }
      "test.source"(%alloc) : (memref<2xf32>) -> ()
      "test.source"(%alloc_0) : (memref<4xf32>) -> ()
    }
    return
}
// CHECK-LABEL: func @nested_forall
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : {alignment = 64 : i64} : memref<640xi8>
//       CHECK: %VIEW = memref.view %[[ALLOC0]][%{{.*}}][] : memref<640xi8> to memref<640xi8>
//       CHECK: %VIEW1= memref.view %[[ALLOC0]][%{{.*}}][] : memref<640xi8> to memref<2xf32>
//  CHECK-NEXT: memref.copy %arg0, %[[VIEW1]]
//  CHECK-NEXT: scf.forall
//       CHECK:   %VIEW2 = memref.view %[[VIEW]][%{{.*}}][] : memref<640xi8> to memref<4xf32>
//       CHECK:   %VIEW3 = memref.view %[[VIEW]][%{{.*}}][] : memref<640xi8> to memref<8xf32>
//  CHECK-NEXT:   scf.forall
//       CHECK:     %VIEW4 = memref.view %[[VIEW]][%{{.*}}][] : memref<640xi8> to memref<16xf32>

func.func @mixed_forall_and_for(
    %lb: index,
    %ub: index,
    %step: index,
    %arg0: memref<2xf32>) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %0 = memref.alloc() : memref<2xf32>
    scf.forall (%i) in (%c16) {
      %alloc = memref.alloc() : memref<4xf32>
      scf.forall (%j) in (%c4) {
        %alloc_0 = memref.alloc() : memref<8xf32>
        %1 = scf.for %k = %lb to %ub step %step
          iter_args(%iterBuf = %arg0) -> (memref<2xf32>) {
          "test.source"(%alloc) : (memref<4xf32>) -> ()
          "test.source"(%alloc_0) : (memref<8xf32>) -> ()
          scf.yield %0 : memref<2xf32>
        }
      }
    }
    return
}
// CHECK-LABEL: func @mixed_forall_and_for
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : {alignment = 64 : i64} : memref<384xi8>
//       CHECK: %VIEW = memref.view %[[ALLOC0]][%{{.*}}][] : memref<384xi8> to memref<384xi8>
//       CHECK: %VIEW1 = memref.view %[[ALLOC0]][%{{.*}}][] : memref<384xi8> to memref<2xf32>
//  CHECK-NEXT: scf.forall
//       CHECK:   %VIEW2 = memref.view %[[VIEW]][%1][] : memref<384xi8> to memref<4xf32>
//  CHECK-NEXT:   scf.forall
//       CHECK:     %VIEW3 = memref.view %[[VIEW]][%{{.*}}][] : memref<384xi8> to memref<8xf32>
//  CHECK-NEXT:     scf.for
//       CHECK:     scf.yield %[[VIEW1]]

func.func @nested_forall_with_dynamic_shape(%arg0: index) {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.forall (%i) in (%c16) {
      %alloc = memref.alloc(%arg0) : memref<?xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc) : (memref<?xf32>) -> ()
      }
    }
    return
}
// CHECK-LABEL: func @nested_forall_with_dynamic_shape
//   CHECK-NOT: memref.alloc
//       CHECK: scf.forall
//  CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc(%arg0) : memref<?xf32>
//  CHECK-NEXT: scf.forall

func.func @nested_forall_with_multi_blocks(
    %lb: index,
    %ub: index,
    %step: index,
    %arg0: memref<2xf32>) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %0 = memref.alloc() : memref<2xf32>
    scf.forall (%i) in (%c16) {
      scf.forall(%j1) in (%c4) {
        %alloc = memref.alloc() : memref<4xf32>
        scf.forall (%k1) in (%c4) {
          %1 = scf.for %kk = %lb to %ub step %step
            iter_args(%iterBuf = %arg0) -> (memref<2xf32>) {
            %alloc_0 = memref.alloc() : memref<2xf32>
            "test.source"(%alloc) : (memref<4xf32>) -> ()
            "test.source"(%alloc_0) : (memref<2xf32>) -> ()
            scf.yield %0 : memref<2xf32>
          }
        }
      }

      scf.forall (%j2) in (%c8) {
        %alloc_2 = memref.alloc() : memref<8xf32>
        scf.forall (%k2) in (%c4) {
          "test.source"(%alloc_2) : (memref<8xf32>) -> ()
        }
      }
    }
    return
}
// CHECK-LABEL: func @nested_forall_with_multi_blocks
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : {alignment = 64 : i64} : memref<4096xi8>
//       CHECK: %VIEW = memref.view %[[ALLOC0]][%{{.*}}][] : memref<4096xi8> to memref<4096xi8>
//       CHECK: %VIEW1 = memref.view %[[ALLOC0]][%{{.*}}][] : memref<4096xi8> to memref<2xf32>
//  CHECK-NEXT: scf.forall
//  CHECK-NEXT:   scf.forall
//   CHECK-NOT:     memref.alloc
//       CHECK:     %VIEW2 = memref.view %[[VIEW]][%{{.*}}][] : memref<4096xi8> to memref<4xf32>
//  CHECK-NEXT:     scf.forall
//   CHECK-NOT:       memref.alloc
//       CHECK:       scf.for
//       CHECK:         %vVIEW3 = memref.view %[[VIEW]][%{{.*}}][] : memref<4096xi8> to memref<2xf32>
//       CHECK:   scf.forall
//       CHECK:     %VIEW4 = memref.view %[[VIEW]][%{{.*}}][] : memref<4096xi8> to memref<8xf32>
//   CHECK-NOT:     memref.alloc
//  CHECK-NEXT:     scf.forall

func.func @nested_forall_nd() {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.forall (%i) in (%c16) {
      %alloc_0 = memref.alloc() : memref<2x4xf32>
      %alloc_1 = memref.alloc() : memref<4x8xf32>
      scf.forall (%j) in (%c4) {
        "test.source"(%alloc_0) : (memref<2x4xf32>) -> ()
        %alloc_2 = memref.alloc() : memref<2x16xf32>
        "test.source"(%alloc_2) : (memref<2x16xf32>) -> ()
      }
      "test.source"(%alloc_1) : (memref<4x8xf32>) -> ()
    }
    return
}
// CHECK-LABEL: func @nested_forall_nd
//       CHECK: %[[ALLOC0:.*]] = memref.alloc() : {alignment = 64 : i64} : memref<640xi8>
//       CHECK: %VIEW = memref.view %[[ALLOC0]][%{{.*}}][] : memref<640xi8> to memref<640xi8>
//       CHECK: scf.forall
//       CHECK:   %VIEW1 = memref.view %[[VIEW]][%1][] : memref<640xi8> to memref<2x4xf32>
//       CHECK:   %VIEW2 = memref.view %[[VIEW]][%{{.*}}][] : memref<640xi8> to memref<4x8xf32>
//  CHECK-NEXT:   scf.forall
//       CHECK:     %VIEW3 = memref.view %[[VIEW]][%{{.*}}][] : memref<640xi8> to memref<2x16xf32>
