
;
XPlaceholder*
dtype0*
shape:ĸĸĸĸĸĸĸĸĸ
7
yPlaceholder*
dtype0	*
shape:ĸĸĸĸĸĸĸĸĸ
H
random_normal/shapeConst*
valueB"      *
dtype0
?
random_normal/meanConst*
valueB
 *    *
dtype0
A
random_normal/stddevConst*
dtype0*
valueB
 *  ?
~
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
T0*
dtype0*
seed2 *

seed 
[
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0
D
random_normalAddrandom_normal/mulrandom_normal/mean*
T0
X
w_h1
VariableV2*
shape
:*
shared_name *
dtype0*
	container 
u
w_h1/AssignAssignw_h1random_normal*
use_locking(*
T0*
_class
	loc:@w_h1*
validate_shape(
=
	w_h1/readIdentityw_h1*
T0*
_class
	loc:@w_h1
J
random_normal_1/shapeConst*
valueB"      *
dtype0
A
random_normal_1/meanConst*
valueB
 *    *
dtype0
C
random_normal_1/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*
seed2 *

seed *
T0
a
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0
J
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0
X
w_h2
VariableV2*
dtype0*
	container *
shape
:*
shared_name 
w
w_h2/AssignAssignw_h2random_normal_1*
use_locking(*
T0*
_class
	loc:@w_h2*
validate_shape(
=
	w_h2/readIdentityw_h2*
T0*
_class
	loc:@w_h2
J
random_normal_2/shapeConst*
valueB"      *
dtype0
A
random_normal_2/meanConst*
valueB
 *    *
dtype0
C
random_normal_2/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*

seed *
T0*
dtype0*
seed2 
a
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0
J
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0
X
w_h3
VariableV2*
shape
:*
shared_name *
dtype0*
	container 
w
w_h3/AssignAssignw_h3random_normal_2*
T0*
_class
	loc:@w_h3*
validate_shape(*
use_locking(
=
	w_h3/readIdentityw_h3*
T0*
_class
	loc:@w_h3
J
random_normal_3/shapeConst*
valueB"      *
dtype0
A
random_normal_3/meanConst*
valueB
 *    *
dtype0
C
random_normal_3/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0
J
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0
X
w_h4
VariableV2*
dtype0*
	container *
shape
:*
shared_name 
w
w_h4/AssignAssignw_h4random_normal_3*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w_h4
=
	w_h4/readIdentityw_h4*
T0*
_class
	loc:@w_h4
J
random_normal_4/shapeConst*
valueB"      *
dtype0
A
random_normal_4/meanConst*
valueB
 *    *
dtype0
C
random_normal_4/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0
J
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0
Y
w_out
VariableV2*
shared_name *
dtype0*
	container *
shape
:
z
w_out/AssignAssignw_outrandom_normal_4*
use_locking(*
T0*
_class

loc:@w_out*
validate_shape(
@

w_out/readIdentityw_out*
T0*
_class

loc:@w_out
C
random_normal_5/shapeConst*
dtype0*
valueB:
A
random_normal_5/meanConst*
valueB
 *    *
dtype0
C
random_normal_5/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0
J
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0
T
b_b1
VariableV2*
shape:*
shared_name *
dtype0*
	container 
w
b_b1/AssignAssignb_b1random_normal_5*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b_b1
=
	b_b1/readIdentityb_b1*
T0*
_class
	loc:@b_b1
C
random_normal_6/shapeConst*
valueB:*
dtype0
A
random_normal_6/meanConst*
valueB
 *    *
dtype0
C
random_normal_6/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*
seed2 *

seed *
T0
a
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0
J
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0
T
b_b2
VariableV2*
dtype0*
	container *
shape:*
shared_name 
w
b_b2/AssignAssignb_b2random_normal_6*
use_locking(*
T0*
_class
	loc:@b_b2*
validate_shape(
=
	b_b2/readIdentityb_b2*
T0*
_class
	loc:@b_b2
C
random_normal_7/shapeConst*
dtype0*
valueB:
A
random_normal_7/meanConst*
valueB
 *    *
dtype0
C
random_normal_7/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0
J
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0
T
b_b3
VariableV2*
dtype0*
	container *
shape:*
shared_name 
w
b_b3/AssignAssignb_b3random_normal_7*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b_b3
=
	b_b3/readIdentityb_b3*
T0*
_class
	loc:@b_b3
C
random_normal_8/shapeConst*
dtype0*
valueB:
A
random_normal_8/meanConst*
valueB
 *    *
dtype0
C
random_normal_8/stddevConst*
valueB
 *  ?*
dtype0

$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*

seed *
T0*
dtype0*
seed2 
a
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0
J
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
T0
T
b_b4
VariableV2*
dtype0*
	container *
shape:*
shared_name 
w
b_b4/AssignAssignb_b4random_normal_8*
use_locking(*
T0*
_class
	loc:@b_b4*
validate_shape(
=
	b_b4/readIdentityb_b4*
T0*
_class
	loc:@b_b4
C
random_normal_9/shapeConst*
valueB:*
dtype0
A
random_normal_9/meanConst*
valueB
 *    *
dtype0
C
random_normal_9/stddevConst*
dtype0*
valueB
 *  ?

$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
T0*
dtype0*
seed2 *

seed 
a
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
T0
J
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0
U
b_out
VariableV2*
dtype0*
	container *
shape:*
shared_name 
z
b_out/AssignAssignb_outrandom_normal_9*
use_locking(*
T0*
_class

loc:@b_out*
validate_shape(
@

b_out/readIdentityb_out*
T0*
_class

loc:@b_out
M
MatMulMatMulX	w_h1/read*
T0*
transpose_a( *
transpose_b( 
&
AddAddMatMul	b_b1/read*
T0

ReluReluAdd*
T0
R
MatMul_1MatMulRelu	w_h2/read*
T0*
transpose_a( *
transpose_b( 
*
Add_1AddMatMul_1	b_b2/read*
T0

Relu_1ReluAdd_1*
T0
T
MatMul_2MatMulRelu_1	w_h3/read*
transpose_a( *
transpose_b( *
T0
*
Add_2AddMatMul_2	b_b3/read*
T0

Relu_2ReluAdd_2*
T0
T
MatMul_3MatMulRelu_2	w_h4/read*
transpose_b( *
T0*
transpose_a( 
*
Add_3AddMatMul_3	b_b4/read*
T0

Relu_3ReluAdd_3*
T0
U
MatMul_4MatMulRelu_2
w_out/read*
transpose_a( *
transpose_b( *
T0
)
addAddMatMul_4
b_out/read*
T0
N
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapey*
T0	*
out_type0

GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsaddy*
T0*
Tlabels0	
3
ConstConst*
dtype0*
valueB: 

lossMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
T0*
	keep_dims( *

Tidx0
H
loss_function/tagsConst*
valueB Bloss_function*
dtype0
A
loss_functionScalarSummaryloss_function/tagsloss*
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
O
!gradients/loss_grad/Reshape/shapeConst*
dtype0*
valueB:
p
gradients/loss_grad/ReshapeReshapegradients/Fill!gradients/loss_grad/Reshape/shape*
T0*
Tshape0

gradients/loss_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0
s
gradients/loss_grad/TileTilegradients/loss_grad/Reshapegradients/loss_grad/Shape*

Tmultiples0*
T0

gradients/loss_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0
D
gradients/loss_grad/Shape_2Const*
valueB *
dtype0
G
gradients/loss_grad/ConstConst*
valueB: *
dtype0
~
gradients/loss_grad/ProdProdgradients/loss_grad/Shape_1gradients/loss_grad/Const*
	keep_dims( *

Tidx0*
T0
I
gradients/loss_grad/Const_1Const*
dtype0*
valueB: 

gradients/loss_grad/Prod_1Prodgradients/loss_grad/Shape_2gradients/loss_grad/Const_1*
	keep_dims( *

Tidx0*
T0
G
gradients/loss_grad/Maximum/yConst*
dtype0*
value	B :
j
gradients/loss_grad/MaximumMaximumgradients/loss_grad/Prod_1gradients/loss_grad/Maximum/y*
T0
h
gradients/loss_grad/floordivFloorDivgradients/loss_grad/Prodgradients/loss_grad/Maximum*
T0
V
gradients/loss_grad/CastCastgradients/loss_grad/floordiv*

DstT0*

SrcT0
c
gradients/loss_grad/truedivRealDivgradients/loss_grad/Tilegradients/loss_grad/Cast*
T0
u
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0

fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*ī
messageĻĨCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()

egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
ĸĸĸĸĸĸĸĸĸ*
dtype0

agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/loss_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0
ĩ
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0
D
gradients/add_grad/ShapeShapeMatMul_4*
T0*
out_type0
H
gradients/add_grad/Shape_1Const*
dtype0*
valueB:

(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
É
gradients/add_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
Í
gradients/add_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
ą
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
·
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1

gradients/MatMul_4_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependency
w_out/read*
T0*
transpose_a( *
transpose_b(

 gradients/MatMul_4_grad/MatMul_1MatMulRelu_2+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
Ã
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul
É
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1
m
gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_4_grad/tuple/control_dependencyRelu_2*
T0
F
gradients/Add_2_grad/ShapeShapeMatMul_2*
T0*
out_type0
J
gradients/Add_2_grad/Shape_1Const*
valueB:*
dtype0

*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0

gradients/Add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
t
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0

gradients/Add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
z
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
đ
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
ŋ
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1

gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependency	w_h3/read*
transpose_a( *
transpose_b(*
T0

 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/Add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
Ã
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul
É
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1
m
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0
F
gradients/Add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0
J
gradients/Add_1_grad/Shape_1Const*
valueB:*
dtype0

*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0

gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
t
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0

gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
z
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
đ
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
ŋ
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1

gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependency	w_h2/read*
transpose_b(*
T0*
transpose_a( 

 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/Add_1_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
Ã
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
É
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
i
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0
B
gradients/Add_grad/ShapeShapeMatMul*
T0*
out_type0
H
gradients/Add_grad/Shape_1Const*
dtype0*
valueB:

(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0

gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
n
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0

gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
t
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
ą
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape
·
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1

gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependency	w_h1/read*
transpose_a( *
transpose_b(*
T0

gradients/MatMul_grad/MatMul_1MatMulX+gradients/Add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ŧ
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
Á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
_
beta1_power/initial_valueConst*
_class
	loc:@b_b1*
valueB
 *fff?*
dtype0
p
beta1_power
VariableV2*
shape: *
shared_name *
_class
	loc:@b_b1*
dtype0*
	container 

beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(
K
beta1_power/readIdentitybeta1_power*
T0*
_class
	loc:@b_b1
_
beta2_power/initial_valueConst*
_class
	loc:@b_b1*
valueB
 *wū?*
dtype0
p
beta2_power
VariableV2*
dtype0*
	container *
shape: *
shared_name *
_class
	loc:@b_b1

beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(
K
beta2_power/readIdentitybeta2_power*
T0*
_class
	loc:@b_b1
y
+w_h1/Adam/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@w_h1*
valueB"      *
dtype0
g
!w_h1/Adam/Initializer/zeros/ConstConst*
dtype0*
_class
	loc:@w_h1*
valueB
 *    
§
w_h1/Adam/Initializer/zerosFill+w_h1/Adam/Initializer/zeros/shape_as_tensor!w_h1/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@w_h1*

index_type0
v
	w_h1/Adam
VariableV2*
shared_name *
_class
	loc:@w_h1*
dtype0*
	container *
shape
:

w_h1/Adam/AssignAssign	w_h1/Adamw_h1/Adam/Initializer/zeros*
T0*
_class
	loc:@w_h1*
validate_shape(*
use_locking(
G
w_h1/Adam/readIdentity	w_h1/Adam*
T0*
_class
	loc:@w_h1
{
-w_h1/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@w_h1*
valueB"      *
dtype0
i
#w_h1/Adam_1/Initializer/zeros/ConstConst*
_class
	loc:@w_h1*
valueB
 *    *
dtype0
­
w_h1/Adam_1/Initializer/zerosFill-w_h1/Adam_1/Initializer/zeros/shape_as_tensor#w_h1/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@w_h1*

index_type0
x
w_h1/Adam_1
VariableV2*
_class
	loc:@w_h1*
dtype0*
	container *
shape
:*
shared_name 

w_h1/Adam_1/AssignAssignw_h1/Adam_1w_h1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@w_h1*
validate_shape(
K
w_h1/Adam_1/readIdentityw_h1/Adam_1*
T0*
_class
	loc:@w_h1
y
+w_h2/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_class
	loc:@w_h2*
valueB"      
g
!w_h2/Adam/Initializer/zeros/ConstConst*
dtype0*
_class
	loc:@w_h2*
valueB
 *    
§
w_h2/Adam/Initializer/zerosFill+w_h2/Adam/Initializer/zeros/shape_as_tensor!w_h2/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@w_h2*

index_type0
v
	w_h2/Adam
VariableV2*
shape
:*
shared_name *
_class
	loc:@w_h2*
dtype0*
	container 

w_h2/Adam/AssignAssign	w_h2/Adamw_h2/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w_h2
G
w_h2/Adam/readIdentity	w_h2/Adam*
T0*
_class
	loc:@w_h2
{
-w_h2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@w_h2*
valueB"      *
dtype0
i
#w_h2/Adam_1/Initializer/zeros/ConstConst*
_class
	loc:@w_h2*
valueB
 *    *
dtype0
­
w_h2/Adam_1/Initializer/zerosFill-w_h2/Adam_1/Initializer/zeros/shape_as_tensor#w_h2/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@w_h2*

index_type0
x
w_h2/Adam_1
VariableV2*
shared_name *
_class
	loc:@w_h2*
dtype0*
	container *
shape
:

w_h2/Adam_1/AssignAssignw_h2/Adam_1w_h2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@w_h2*
validate_shape(
K
w_h2/Adam_1/readIdentityw_h2/Adam_1*
T0*
_class
	loc:@w_h2
y
+w_h3/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_class
	loc:@w_h3*
valueB"      
g
!w_h3/Adam/Initializer/zeros/ConstConst*
_class
	loc:@w_h3*
valueB
 *    *
dtype0
§
w_h3/Adam/Initializer/zerosFill+w_h3/Adam/Initializer/zeros/shape_as_tensor!w_h3/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@w_h3*

index_type0
v
	w_h3/Adam
VariableV2*
shape
:*
shared_name *
_class
	loc:@w_h3*
dtype0*
	container 

w_h3/Adam/AssignAssign	w_h3/Adamw_h3/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w_h3
G
w_h3/Adam/readIdentity	w_h3/Adam*
T0*
_class
	loc:@w_h3
{
-w_h3/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@w_h3*
valueB"      *
dtype0
i
#w_h3/Adam_1/Initializer/zeros/ConstConst*
_class
	loc:@w_h3*
valueB
 *    *
dtype0
­
w_h3/Adam_1/Initializer/zerosFill-w_h3/Adam_1/Initializer/zeros/shape_as_tensor#w_h3/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@w_h3*

index_type0
x
w_h3/Adam_1
VariableV2*
_class
	loc:@w_h3*
dtype0*
	container *
shape
:*
shared_name 

w_h3/Adam_1/AssignAssignw_h3/Adam_1w_h3/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w_h3
K
w_h3/Adam_1/readIdentityw_h3/Adam_1*
T0*
_class
	loc:@w_h3
{
,w_out/Adam/Initializer/zeros/shape_as_tensorConst*
_class

loc:@w_out*
valueB"      *
dtype0
i
"w_out/Adam/Initializer/zeros/ConstConst*
_class

loc:@w_out*
valueB
 *    *
dtype0
Ŧ
w_out/Adam/Initializer/zerosFill,w_out/Adam/Initializer/zeros/shape_as_tensor"w_out/Adam/Initializer/zeros/Const*
T0*
_class

loc:@w_out*

index_type0
x

w_out/Adam
VariableV2*
shape
:*
shared_name *
_class

loc:@w_out*
dtype0*
	container 

w_out/Adam/AssignAssign
w_out/Adamw_out/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class

loc:@w_out
J
w_out/Adam/readIdentity
w_out/Adam*
T0*
_class

loc:@w_out
}
.w_out/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class

loc:@w_out*
valueB"      *
dtype0
k
$w_out/Adam_1/Initializer/zeros/ConstConst*
_class

loc:@w_out*
valueB
 *    *
dtype0
ą
w_out/Adam_1/Initializer/zerosFill.w_out/Adam_1/Initializer/zeros/shape_as_tensor$w_out/Adam_1/Initializer/zeros/Const*
T0*
_class

loc:@w_out*

index_type0
z
w_out/Adam_1
VariableV2*
dtype0*
	container *
shape
:*
shared_name *
_class

loc:@w_out

w_out/Adam_1/AssignAssignw_out/Adam_1w_out/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class

loc:@w_out
N
w_out/Adam_1/readIdentityw_out/Adam_1*
T0*
_class

loc:@w_out
r
+b_b1/Adam/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@b_b1*
valueB:*
dtype0
g
!b_b1/Adam/Initializer/zeros/ConstConst*
_class
	loc:@b_b1*
valueB
 *    *
dtype0
§
b_b1/Adam/Initializer/zerosFill+b_b1/Adam/Initializer/zeros/shape_as_tensor!b_b1/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@b_b1*

index_type0
r
	b_b1/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class
	loc:@b_b1

b_b1/Adam/AssignAssign	b_b1/Adamb_b1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(
G
b_b1/Adam/readIdentity	b_b1/Adam*
T0*
_class
	loc:@b_b1
t
-b_b1/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@b_b1*
valueB:*
dtype0
i
#b_b1/Adam_1/Initializer/zeros/ConstConst*
_class
	loc:@b_b1*
valueB
 *    *
dtype0
­
b_b1/Adam_1/Initializer/zerosFill-b_b1/Adam_1/Initializer/zeros/shape_as_tensor#b_b1/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@b_b1*

index_type0
t
b_b1/Adam_1
VariableV2*
_class
	loc:@b_b1*
dtype0*
	container *
shape:*
shared_name 

b_b1/Adam_1/AssignAssignb_b1/Adam_1b_b1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(
K
b_b1/Adam_1/readIdentityb_b1/Adam_1*
T0*
_class
	loc:@b_b1
r
+b_b2/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_class
	loc:@b_b2*
valueB:
g
!b_b2/Adam/Initializer/zeros/ConstConst*
dtype0*
_class
	loc:@b_b2*
valueB
 *    
§
b_b2/Adam/Initializer/zerosFill+b_b2/Adam/Initializer/zeros/shape_as_tensor!b_b2/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@b_b2*

index_type0
r
	b_b2/Adam
VariableV2*
_class
	loc:@b_b2*
dtype0*
	container *
shape:*
shared_name 

b_b2/Adam/AssignAssign	b_b2/Adamb_b2/Adam/Initializer/zeros*
T0*
_class
	loc:@b_b2*
validate_shape(*
use_locking(
G
b_b2/Adam/readIdentity	b_b2/Adam*
T0*
_class
	loc:@b_b2
t
-b_b2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@b_b2*
valueB:*
dtype0
i
#b_b2/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_class
	loc:@b_b2*
valueB
 *    
­
b_b2/Adam_1/Initializer/zerosFill-b_b2/Adam_1/Initializer/zeros/shape_as_tensor#b_b2/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@b_b2*

index_type0
t
b_b2/Adam_1
VariableV2*
shape:*
shared_name *
_class
	loc:@b_b2*
dtype0*
	container 

b_b2/Adam_1/AssignAssignb_b2/Adam_1b_b2/Adam_1/Initializer/zeros*
T0*
_class
	loc:@b_b2*
validate_shape(*
use_locking(
K
b_b2/Adam_1/readIdentityb_b2/Adam_1*
T0*
_class
	loc:@b_b2
r
+b_b3/Adam/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@b_b3*
valueB:*
dtype0
g
!b_b3/Adam/Initializer/zeros/ConstConst*
_class
	loc:@b_b3*
valueB
 *    *
dtype0
§
b_b3/Adam/Initializer/zerosFill+b_b3/Adam/Initializer/zeros/shape_as_tensor!b_b3/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@b_b3*

index_type0
r
	b_b3/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class
	loc:@b_b3

b_b3/Adam/AssignAssign	b_b3/Adamb_b3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_b3*
validate_shape(
G
b_b3/Adam/readIdentity	b_b3/Adam*
T0*
_class
	loc:@b_b3
t
-b_b3/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@b_b3*
valueB:*
dtype0
i
#b_b3/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_class
	loc:@b_b3*
valueB
 *    
­
b_b3/Adam_1/Initializer/zerosFill-b_b3/Adam_1/Initializer/zeros/shape_as_tensor#b_b3/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@b_b3*

index_type0
t
b_b3/Adam_1
VariableV2*
shared_name *
_class
	loc:@b_b3*
dtype0*
	container *
shape:

b_b3/Adam_1/AssignAssignb_b3/Adam_1b_b3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_b3*
validate_shape(
K
b_b3/Adam_1/readIdentityb_b3/Adam_1*
T0*
_class
	loc:@b_b3
t
,b_out/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_class

loc:@b_out*
valueB:
i
"b_out/Adam/Initializer/zeros/ConstConst*
dtype0*
_class

loc:@b_out*
valueB
 *    
Ŧ
b_out/Adam/Initializer/zerosFill,b_out/Adam/Initializer/zeros/shape_as_tensor"b_out/Adam/Initializer/zeros/Const*
T0*
_class

loc:@b_out*

index_type0
t

b_out/Adam
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_class

loc:@b_out

b_out/Adam/AssignAssign
b_out/Adamb_out/Adam/Initializer/zeros*
use_locking(*
T0*
_class

loc:@b_out*
validate_shape(
J
b_out/Adam/readIdentity
b_out/Adam*
T0*
_class

loc:@b_out
v
.b_out/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class

loc:@b_out*
valueB:*
dtype0
k
$b_out/Adam_1/Initializer/zeros/ConstConst*
_class

loc:@b_out*
valueB
 *    *
dtype0
ą
b_out/Adam_1/Initializer/zerosFill.b_out/Adam_1/Initializer/zeros/shape_as_tensor$b_out/Adam_1/Initializer/zeros/Const*
T0*
_class

loc:@b_out*

index_type0
v
b_out/Adam_1
VariableV2*
shared_name *
_class

loc:@b_out*
dtype0*
	container *
shape:

b_out/Adam_1/AssignAssignb_out/Adam_1b_out/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class

loc:@b_out
N
b_out/Adam_1/readIdentityb_out/Adam_1*
T0*
_class

loc:@b_out
?
Adam/learning_rateConst*
valueB
 *·Ņ8*
dtype0
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
dtype0*
valueB
 *wū?
9
Adam/epsilonConst*
dtype0*
valueB
 *wĖ+2

Adam/update_w_h1/ApplyAdam	ApplyAdamw_h1	w_h1/Adamw_h1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_class
	loc:@w_h1
 
Adam/update_w_h2/ApplyAdam	ApplyAdamw_h2	w_h2/Adamw_h2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_class
	loc:@w_h2
 
Adam/update_w_h3/ApplyAdam	ApplyAdamw_h3	w_h3/Adamw_h3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
T0*
_class
	loc:@w_h3*
use_nesterov( *
use_locking( 
Ĩ
Adam/update_w_out/ApplyAdam	ApplyAdamw_out
w_out/Adamw_out/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_4_grad/tuple/control_dependency_1*
T0*
_class

loc:@w_out*
use_nesterov( *
use_locking( 

Adam/update_b_b1/ApplyAdam	ApplyAdamb_b1	b_b1/Adamb_b1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
T0*
_class
	loc:@b_b1

Adam/update_b_b2/ApplyAdam	ApplyAdamb_b2	b_b2/Adamb_b2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b_b2*
use_nesterov( 

Adam/update_b_b3/ApplyAdam	ApplyAdamb_b3	b_b3/Adamb_b3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b_b3*
use_nesterov( 
 
Adam/update_b_out/ApplyAdam	ApplyAdamb_out
b_out/Adamb_out/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class

loc:@b_out*
use_nesterov( *
use_locking( 
đ
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_w_h1/ApplyAdam^Adam/update_w_h2/ApplyAdam^Adam/update_w_h3/ApplyAdam^Adam/update_w_out/ApplyAdam^Adam/update_b_b1/ApplyAdam^Adam/update_b_b2/ApplyAdam^Adam/update_b_b3/ApplyAdam^Adam/update_b_out/ApplyAdam*
T0*
_class
	loc:@b_b1
w
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
	loc:@b_b1*
validate_shape(
ŧ

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_w_h1/ApplyAdam^Adam/update_w_h2/ApplyAdam^Adam/update_w_h3/ApplyAdam^Adam/update_w_out/ApplyAdam^Adam/update_b_b1/ApplyAdam^Adam/update_b_b2/ApplyAdam^Adam/update_b_b3/ApplyAdam^Adam/update_b_out/ApplyAdam*
T0*
_class
	loc:@b_b1
{
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
	loc:@b_b1*
validate_shape(*
use_locking( 

AdamNoOp^Adam/update_w_h1/ApplyAdam^Adam/update_w_h2/ApplyAdam^Adam/update_w_h3/ApplyAdam^Adam/update_w_out/ApplyAdam^Adam/update_b_b1/ApplyAdam^Adam/update_b_b2/ApplyAdam^Adam/update_b_b3/ApplyAdam^Adam/update_b_out/ApplyAdam^Adam/Assign^Adam/Assign_1
 
SoftmaxSoftmaxadd*
T0
:
y_pred/dimensionConst*
value	B :*
dtype0
S
y_predArgMaxSoftmaxy_pred/dimension*

Tidx0*
T0*
output_type0	
/
correct_predictionEqualy_predy*
T0	
8
CastCastcorrect_prediction*

DstT0*

SrcT0

5
Const_1Const*
valueB: *
dtype0
E
accuracyMeanCastConst_1*
T0*
	keep_dims( *

Tidx0
P
accuracy_function/tagsConst*"
valueB Baccuracy_function*
dtype0
M
accuracy_functionScalarSummaryaccuracy_function/tagsaccuracy*
T0
8

save/ConstConst*
dtype0*
valueB Bmodel
â
save/SaveV2/tensor_namesConst*ą
value§BĪBb_b1B	b_b1/AdamBb_b1/Adam_1Bb_b2B	b_b2/AdamBb_b2/Adam_1Bb_b3B	b_b3/AdamBb_b3/Adam_1Bb_b4Bb_outB
b_out/AdamBb_out/Adam_1Bbeta1_powerBbeta2_powerBw_h1B	w_h1/AdamBw_h1/Adam_1Bw_h2B	w_h2/AdamBw_h2/Adam_1Bw_h3B	w_h3/AdamBw_h3/Adam_1Bw_h4Bw_outB
w_out/AdamBw_out/Adam_1*
dtype0

save/SaveV2/shape_and_slicesConst*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ą
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb_b1	b_b1/Adamb_b1/Adam_1b_b2	b_b2/Adamb_b2/Adam_1b_b3	b_b3/Adamb_b3/Adam_1b_b4b_out
b_out/Adamb_out/Adam_1beta1_powerbeta2_powerw_h1	w_h1/Adamw_h1/Adam_1w_h2	w_h2/Adamw_h2/Adam_1w_h3	w_h3/Adamw_h3/Adam_1w_h4w_out
w_out/Adamw_out/Adam_1**
dtypes 
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
ô
save/RestoreV2/tensor_namesConst"/device:CPU:0*ą
value§BĪBb_b1B	b_b1/AdamBb_b1/Adam_1Bb_b2B	b_b2/AdamBb_b2/Adam_1Bb_b3B	b_b3/AdamBb_b3/Adam_1Bb_b4Bb_outB
b_out/AdamBb_out/Adam_1Bbeta1_powerBbeta2_powerBw_h1B	w_h1/AdamBw_h1/Adam_1Bw_h2B	w_h2/AdamBw_h2/Adam_1Bw_h3B	w_h3/AdamBw_h3/Adam_1Bw_h4Bw_outB
w_out/AdamBw_out/Adam_1*
dtype0

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
 
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0**
dtypes 
2
v
save/AssignAssignb_b1save/RestoreV2*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(

save/Assign_1Assign	b_b1/Adamsave/RestoreV2:1*
T0*
_class
	loc:@b_b1*
validate_shape(*
use_locking(

save/Assign_2Assignb_b1/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(
z
save/Assign_3Assignb_b2save/RestoreV2:3*
use_locking(*
T0*
_class
	loc:@b_b2*
validate_shape(

save/Assign_4Assign	b_b2/Adamsave/RestoreV2:4*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b_b2

save/Assign_5Assignb_b2/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
	loc:@b_b2*
validate_shape(
z
save/Assign_6Assignb_b3save/RestoreV2:6*
T0*
_class
	loc:@b_b3*
validate_shape(*
use_locking(

save/Assign_7Assign	b_b3/Adamsave/RestoreV2:7*
use_locking(*
T0*
_class
	loc:@b_b3*
validate_shape(

save/Assign_8Assignb_b3/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
	loc:@b_b3*
validate_shape(
z
save/Assign_9Assignb_b4save/RestoreV2:9*
validate_shape(*
use_locking(*
T0*
_class
	loc:@b_b4
~
save/Assign_10Assignb_outsave/RestoreV2:10*
use_locking(*
T0*
_class

loc:@b_out*
validate_shape(

save/Assign_11Assign
b_out/Adamsave/RestoreV2:11*
use_locking(*
T0*
_class

loc:@b_out*
validate_shape(

save/Assign_12Assignb_out/Adam_1save/RestoreV2:12*
use_locking(*
T0*
_class

loc:@b_out*
validate_shape(

save/Assign_13Assignbeta1_powersave/RestoreV2:13*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(

save/Assign_14Assignbeta2_powersave/RestoreV2:14*
use_locking(*
T0*
_class
	loc:@b_b1*
validate_shape(
|
save/Assign_15Assignw_h1save/RestoreV2:15*
use_locking(*
T0*
_class
	loc:@w_h1*
validate_shape(

save/Assign_16Assign	w_h1/Adamsave/RestoreV2:16*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w_h1

save/Assign_17Assignw_h1/Adam_1save/RestoreV2:17*
use_locking(*
T0*
_class
	loc:@w_h1*
validate_shape(
|
save/Assign_18Assignw_h2save/RestoreV2:18*
use_locking(*
T0*
_class
	loc:@w_h2*
validate_shape(

save/Assign_19Assign	w_h2/Adamsave/RestoreV2:19*
T0*
_class
	loc:@w_h2*
validate_shape(*
use_locking(

save/Assign_20Assignw_h2/Adam_1save/RestoreV2:20*
T0*
_class
	loc:@w_h2*
validate_shape(*
use_locking(
|
save/Assign_21Assignw_h3save/RestoreV2:21*
validate_shape(*
use_locking(*
T0*
_class
	loc:@w_h3

save/Assign_22Assign	w_h3/Adamsave/RestoreV2:22*
use_locking(*
T0*
_class
	loc:@w_h3*
validate_shape(

save/Assign_23Assignw_h3/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
	loc:@w_h3*
validate_shape(
|
save/Assign_24Assignw_h4save/RestoreV2:24*
use_locking(*
T0*
_class
	loc:@w_h4*
validate_shape(
~
save/Assign_25Assignw_outsave/RestoreV2:25*
T0*
_class

loc:@w_out*
validate_shape(*
use_locking(

save/Assign_26Assign
w_out/Adamsave/RestoreV2:26*
validate_shape(*
use_locking(*
T0*
_class

loc:@w_out

save/Assign_27Assignw_out/Adam_1save/RestoreV2:27*
T0*
_class

loc:@w_out*
validate_shape(*
use_locking(
č
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27
M
Merge/MergeSummaryMergeSummaryloss_functionaccuracy_function*
N

initNoOp^w_h1/Assign^w_h2/Assign^w_h3/Assign^w_h4/Assign^w_out/Assign^b_b1/Assign^b_b2/Assign^b_b3/Assign^b_b4/Assign^b_out/Assign^beta1_power/Assign^beta2_power/Assign^w_h1/Adam/Assign^w_h1/Adam_1/Assign^w_h2/Adam/Assign^w_h2/Adam_1/Assign^w_h3/Adam/Assign^w_h3/Adam_1/Assign^w_out/Adam/Assign^w_out/Adam_1/Assign^b_b1/Adam/Assign^b_b1/Adam_1/Assign^b_b2/Adam/Assign^b_b2/Adam_1/Assign^b_b3/Adam/Assign^b_b3/Adam_1/Assign^b_out/Adam/Assign^b_out/Adam_1/Assign"