
;
XPlaceholder*
dtype0*
shape:���������
�
w_h1Const*
dtype0*�
value�B�"�6:�=�A'��� ��YG�G	�?���,��\ĸ?>T? ��T� ?#�Y>�,�'5>u�)?/��?,Wʾt'�FZD�D�5@�.��T0??Q@@|?1T�}�~?�h�>��]���>�������=�O�������Z>�p?�M���\�?�;S?q�>b�;��A?n/�?��4?�u��=<�>�m4?b+���6?
=
	w_h1/readIdentityw_h1*
T0*
_class
	loc:@w_h1
�
w_h2Const*�
value�B�"����?�>�=��I�7;�ʹ���>�ξf��?���4����k>���?���E䐿��?)b=?��C>e���F��^�>k#ԾW�����?� �=pz
?{�ʽ�@�O��fL�@~�����|!���/7<�	9��C?�Cf��CS��R?�7�>.>��:��F�)�`��˰?*a���;?Q�~��5�>*
dtype0
=
	w_h2/readIdentityw_h2*
T0*
_class
	loc:@w_h2
�
w_h3Const*y
valuepBn"`)�?P�@��=�ς?1��>���i�ο��a>�a���K���]?D��?� ?� d?��@3����?A��ü�����|�o?{FK?����W��?*
dtype0
=
	w_h3/readIdentityw_h3*
T0*
_class
	loc:@w_h3
V
w_outConst*
dtype0*9
value0B." �e���ht��#���"�=��?���>�ߥ?= N?
@

w_out/readIdentityw_out*
T0*
_class

loc:@w_out
Q
b_b1Const*5
value,B*" e@�?���?$i �3�A?Dm�<r?��?Gx��*
dtype0
=
	b_b1/readIdentityb_b1*
T0*
_class
	loc:@b_b1
I
b_b2Const*-
value$B""nz!?ZeP>p	�?O##>w;?��'?*
dtype0
=
	b_b2/readIdentityb_b2*
T0*
_class
	loc:@b_b2
A
b_b3Const*%
valueB"|����[�����#)?*
dtype0
=
	b_b3/readIdentityb_b3*
T0*
_class
	loc:@b_b3
:
b_outConst*
dtype0*
valueB"(�Ѽ��T�
@

b_out/readIdentityb_out*
T0*
_class

loc:@b_out
M
MatMulMatMulX	w_h1/read*
transpose_b( *
T0*
transpose_a( 
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
MatMul_2MatMulRelu_1	w_h3/read*
T0*
transpose_a( *
transpose_b( 
*
Add_2AddMatMul_2	b_b3/read*
T0

Relu_2ReluAdd_2*
T0
U
MatMul_4MatMulRelu_2
w_out/read*
T0*
transpose_a( *
transpose_b( 
)
addAddMatMul_4
b_out/read*
T0
 
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
output_type0	 