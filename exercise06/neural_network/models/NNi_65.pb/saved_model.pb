??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8Ռ
|
dense_168/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_168/kernel
u
$dense_168/kernel/Read/ReadVariableOpReadVariableOpdense_168/kernel*
_output_shapes

:*
dtype0
t
dense_168/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_168/bias
m
"dense_168/bias/Read/ReadVariableOpReadVariableOpdense_168/bias*
_output_shapes
:*
dtype0
|
dense_169/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_169/kernel
u
$dense_169/kernel/Read/ReadVariableOpReadVariableOpdense_169/kernel*
_output_shapes

:*
dtype0
t
dense_169/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_169/bias
m
"dense_169/bias/Read/ReadVariableOpReadVariableOpdense_169/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_168/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_168/kernel/m
?
+Adam/dense_168/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_168/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_168/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_168/bias/m
{
)Adam/dense_168/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_168/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_169/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_169/kernel/m
?
+Adam/dense_169/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_169/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_169/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_169/bias/m
{
)Adam/dense_169/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_169/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_168/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_168/kernel/v
?
+Adam/dense_168/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_168/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_168/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_168/bias/v
{
)Adam/dense_168/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_168/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_169/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_169/kernel/v
?
+Adam/dense_169/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_169/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_169/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_169/bias/v
{
)Adam/dense_169/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_169/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_rate	m.
m/m0m1	v2
v3v4v5
 

	0

1
2
3

	0

1
2
3
?
layer_metrics
regularization_losses
non_trainable_variables
layer_regularization_losses
trainable_variables
	variables

layers
metrics
 
\Z
VARIABLE_VALUEdense_168/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_168/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
?
layer_metrics
regularization_losses
 non_trainable_variables
!layer_regularization_losses
trainable_variables
	variables

"layers
#metrics
\Z
VARIABLE_VALUEdense_169/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_169/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
$layer_metrics
regularization_losses
%non_trainable_variables
&layer_regularization_losses
trainable_variables
	variables

'layers
(metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1

)0
 
 
 
 
 
 
 
 
 
 
4
	*total
	+count
,	variables
-	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

,	variables
}
VARIABLE_VALUEAdam/dense_168/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_168/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_169/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_169/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_168/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_168/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_169/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_169/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_65Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_65dense_168/kerneldense_168/biasdense_169/kerneldense_169/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1321386
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_168/kernel/Read/ReadVariableOp"dense_168/bias/Read/ReadVariableOp$dense_169/kernel/Read/ReadVariableOp"dense_169/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_168/kernel/m/Read/ReadVariableOp)Adam/dense_168/bias/m/Read/ReadVariableOp+Adam/dense_169/kernel/m/Read/ReadVariableOp)Adam/dense_169/bias/m/Read/ReadVariableOp+Adam/dense_168/kernel/v/Read/ReadVariableOp)Adam/dense_168/bias/v/Read/ReadVariableOp+Adam/dense_169/kernel/v/Read/ReadVariableOp)Adam/dense_169/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1321565
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_168/kerneldense_168/biasdense_169/kerneldense_169/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_168/kernel/mAdam/dense_168/bias/mAdam/dense_169/kernel/mAdam/dense_169/bias/mAdam/dense_168/kernel/vAdam/dense_168/bias/vAdam/dense_169/kernel/vAdam/dense_169/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1321632??
?
?
/__inference_sequential_64_layer_call_fn_1321336
input_65
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_65unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_13213252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_65
?
?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321325

inputs
dense_168_1321314
dense_168_1321316
dense_169_1321319
dense_169_1321321
identity??!dense_168/StatefulPartitionedCall?!dense_169/StatefulPartitionedCall?
!dense_168/StatefulPartitionedCallStatefulPartitionedCallinputsdense_168_1321314dense_168_1321316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_168_layer_call_and_return_conditional_losses_13212512#
!dense_168/StatefulPartitionedCall?
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_1321319dense_169_1321321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_169_layer_call_and_return_conditional_losses_13212772#
!dense_169/StatefulPartitionedCall?
IdentityIdentity*dense_169/StatefulPartitionedCall:output:0"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321403

inputs,
(dense_168_matmul_readvariableop_resource-
)dense_168_biasadd_readvariableop_resource,
(dense_169_matmul_readvariableop_resource-
)dense_169_biasadd_readvariableop_resource
identity??
dense_168/MatMul/ReadVariableOpReadVariableOp(dense_168_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_168/MatMul/ReadVariableOp?
dense_168/MatMulMatMulinputs'dense_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_168/MatMul?
 dense_168/BiasAdd/ReadVariableOpReadVariableOp)dense_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_168/BiasAdd/ReadVariableOp?
dense_168/BiasAddBiasAdddense_168/MatMul:product:0(dense_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_168/BiasAdd
dense_168/SigmoidSigmoiddense_168/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_168/Sigmoid?
dense_169/MatMul/ReadVariableOpReadVariableOp(dense_169_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_169/MatMul/ReadVariableOp?
dense_169/MatMulMatMuldense_168/Sigmoid:y:0'dense_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_169/MatMul?
 dense_169/BiasAdd/ReadVariableOpReadVariableOp)dense_169_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_169/BiasAdd/ReadVariableOp?
dense_169/BiasAddBiasAdddense_169/MatMul:product:0(dense_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_169/BiasAddn
IdentityIdentitydense_169/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_1321236
input_65:
6sequential_64_dense_168_matmul_readvariableop_resource;
7sequential_64_dense_168_biasadd_readvariableop_resource:
6sequential_64_dense_169_matmul_readvariableop_resource;
7sequential_64_dense_169_biasadd_readvariableop_resource
identity??
-sequential_64/dense_168/MatMul/ReadVariableOpReadVariableOp6sequential_64_dense_168_matmul_readvariableop_resource*
_output_shapes

:*
dtype02/
-sequential_64/dense_168/MatMul/ReadVariableOp?
sequential_64/dense_168/MatMulMatMulinput_655sequential_64/dense_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_64/dense_168/MatMul?
.sequential_64/dense_168/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_dense_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_64/dense_168/BiasAdd/ReadVariableOp?
sequential_64/dense_168/BiasAddBiasAdd(sequential_64/dense_168/MatMul:product:06sequential_64/dense_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_64/dense_168/BiasAdd?
sequential_64/dense_168/SigmoidSigmoid(sequential_64/dense_168/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2!
sequential_64/dense_168/Sigmoid?
-sequential_64/dense_169/MatMul/ReadVariableOpReadVariableOp6sequential_64_dense_169_matmul_readvariableop_resource*
_output_shapes

:*
dtype02/
-sequential_64/dense_169/MatMul/ReadVariableOp?
sequential_64/dense_169/MatMulMatMul#sequential_64/dense_168/Sigmoid:y:05sequential_64/dense_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_64/dense_169/MatMul?
.sequential_64/dense_169/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_dense_169_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_64/dense_169/BiasAdd/ReadVariableOp?
sequential_64/dense_169/BiasAddBiasAdd(sequential_64/dense_169/MatMul:product:06sequential_64/dense_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_64/dense_169/BiasAdd|
IdentityIdentity(sequential_64/dense_169/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_65
?0
?
 __inference__traced_save_1321565
file_prefix/
+savev2_dense_168_kernel_read_readvariableop-
)savev2_dense_168_bias_read_readvariableop/
+savev2_dense_169_kernel_read_readvariableop-
)savev2_dense_169_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_168_kernel_m_read_readvariableop4
0savev2_adam_dense_168_bias_m_read_readvariableop6
2savev2_adam_dense_169_kernel_m_read_readvariableop4
0savev2_adam_dense_169_bias_m_read_readvariableop6
2savev2_adam_dense_168_kernel_v_read_readvariableop4
0savev2_adam_dense_168_bias_v_read_readvariableop6
2savev2_adam_dense_169_kernel_v_read_readvariableop4
0savev2_adam_dense_169_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_23ed561d16e04261a17194672a61f9b7/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_168_kernel_read_readvariableop)savev2_dense_168_bias_read_readvariableop+savev2_dense_169_kernel_read_readvariableop)savev2_dense_169_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_168_kernel_m_read_readvariableop0savev2_adam_dense_168_bias_m_read_readvariableop2savev2_adam_dense_169_kernel_m_read_readvariableop0savev2_adam_dense_169_bias_m_read_readvariableop2savev2_adam_dense_168_kernel_v_read_readvariableop0savev2_adam_dense_168_bias_v_read_readvariableop2savev2_adam_dense_169_kernel_v_read_readvariableop0savev2_adam_dense_169_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapest
r: ::::: : : : : : : ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
F__inference_dense_169_layer_call_and_return_conditional_losses_1321277

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_168_layer_call_and_return_conditional_losses_1321457

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_168_layer_call_fn_1321466

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_168_layer_call_and_return_conditional_losses_13212512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?R
?	
#__inference__traced_restore_1321632
file_prefix%
!assignvariableop_dense_168_kernel%
!assignvariableop_1_dense_168_bias'
#assignvariableop_2_dense_169_kernel%
!assignvariableop_3_dense_169_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count/
+assignvariableop_11_adam_dense_168_kernel_m-
)assignvariableop_12_adam_dense_168_bias_m/
+assignvariableop_13_adam_dense_169_kernel_m-
)assignvariableop_14_adam_dense_169_bias_m/
+assignvariableop_15_adam_dense_168_kernel_v-
)assignvariableop_16_adam_dense_168_bias_v/
+assignvariableop_17_adam_dense_169_kernel_v-
)assignvariableop_18_adam_dense_169_bias_v
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_168_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_168_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_169_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_169_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_adam_dense_168_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_dense_168_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_dense_169_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_169_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_dense_168_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_168_bias_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dense_169_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_dense_169_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19?
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
%__inference_signature_wrapper_1321386
input_65
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_65unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_13212362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_65
?
?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321420

inputs,
(dense_168_matmul_readvariableop_resource-
)dense_168_biasadd_readvariableop_resource,
(dense_169_matmul_readvariableop_resource-
)dense_169_biasadd_readvariableop_resource
identity??
dense_168/MatMul/ReadVariableOpReadVariableOp(dense_168_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_168/MatMul/ReadVariableOp?
dense_168/MatMulMatMulinputs'dense_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_168/MatMul?
 dense_168/BiasAdd/ReadVariableOpReadVariableOp)dense_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_168/BiasAdd/ReadVariableOp?
dense_168/BiasAddBiasAdddense_168/MatMul:product:0(dense_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_168/BiasAdd
dense_168/SigmoidSigmoiddense_168/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_168/Sigmoid?
dense_169/MatMul/ReadVariableOpReadVariableOp(dense_169_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_169/MatMul/ReadVariableOp?
dense_169/MatMulMatMuldense_168/Sigmoid:y:0'dense_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_169/MatMul?
 dense_169/BiasAdd/ReadVariableOpReadVariableOp)dense_169_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_169/BiasAdd/ReadVariableOp?
dense_169/BiasAddBiasAdddense_169/MatMul:product:0(dense_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_169/BiasAddn
IdentityIdentitydense_169/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_64_layer_call_fn_1321446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_13213522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_169_layer_call_fn_1321485

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_169_layer_call_and_return_conditional_losses_13212772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321352

inputs
dense_168_1321341
dense_168_1321343
dense_169_1321346
dense_169_1321348
identity??!dense_168/StatefulPartitionedCall?!dense_169/StatefulPartitionedCall?
!dense_168/StatefulPartitionedCallStatefulPartitionedCallinputsdense_168_1321341dense_168_1321343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_168_layer_call_and_return_conditional_losses_13212512#
!dense_168/StatefulPartitionedCall?
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_1321346dense_169_1321348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_169_layer_call_and_return_conditional_losses_13212772#
!dense_169/StatefulPartitionedCall?
IdentityIdentity*dense_169/StatefulPartitionedCall:output:0"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_64_layer_call_fn_1321363
input_65
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_65unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_13213522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_65
?
?
F__inference_dense_168_layer_call_and_return_conditional_losses_1321251

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_64_layer_call_fn_1321433

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_64_layer_call_and_return_conditional_losses_13213252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_169_layer_call_and_return_conditional_losses_1321476

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321294
input_65
dense_168_1321262
dense_168_1321264
dense_169_1321288
dense_169_1321290
identity??!dense_168/StatefulPartitionedCall?!dense_169/StatefulPartitionedCall?
!dense_168/StatefulPartitionedCallStatefulPartitionedCallinput_65dense_168_1321262dense_168_1321264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_168_layer_call_and_return_conditional_losses_13212512#
!dense_168/StatefulPartitionedCall?
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_1321288dense_169_1321290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_169_layer_call_and_return_conditional_losses_13212772#
!dense_169/StatefulPartitionedCall?
IdentityIdentity*dense_169/StatefulPartitionedCall:output:0"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_65
?
?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321308
input_65
dense_168_1321297
dense_168_1321299
dense_169_1321302
dense_169_1321304
identity??!dense_168/StatefulPartitionedCall?!dense_169/StatefulPartitionedCall?
!dense_168/StatefulPartitionedCallStatefulPartitionedCallinput_65dense_168_1321297dense_168_1321299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_168_layer_call_and_return_conditional_losses_13212512#
!dense_168/StatefulPartitionedCall?
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_1321302dense_169_1321304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_169_layer_call_and_return_conditional_losses_13212772#
!dense_169/StatefulPartitionedCall?
IdentityIdentity*dense_169/StatefulPartitionedCall:output:0"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_65"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_651
serving_default_input_65:0?????????=
	dense_1690
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?c
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*6&call_and_return_all_conditional_losses
7__call__
8_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_64", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 21]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_65"}}, {"class_name": "Dense", "config": {"name": "dense_168", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_169", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_64", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 21]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_65"}}, {"class_name": "Dense", "config": {"name": "dense_168", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_169", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
*9&call_and_return_all_conditional_losses
:__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_168", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_168", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 21}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*;&call_and_return_all_conditional_losses
<__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_169", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_169", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
iter

beta_1

beta_2
	decay
learning_rate	m.
m/m0m1	v2
v3v4v5"
	optimizer
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
?
layer_metrics
regularization_losses
non_trainable_variables
layer_regularization_losses
trainable_variables
	variables

layers
metrics
7__call__
8_default_save_signature
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
,
=serving_default"
signature_map
": 2dense_168/kernel
:2dense_168/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?
layer_metrics
regularization_losses
 non_trainable_variables
!layer_regularization_losses
trainable_variables
	variables

"layers
#metrics
:__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
": 2dense_169/kernel
:2dense_169/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
$layer_metrics
regularization_losses
%non_trainable_variables
&layer_regularization_losses
trainable_variables
	variables

'layers
(metrics
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	*total
	+count
,	variables
-	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
*0
+1"
trackable_list_wrapper
-
,	variables"
_generic_user_object
':%2Adam/dense_168/kernel/m
!:2Adam/dense_168/bias/m
':%2Adam/dense_169/kernel/m
!:2Adam/dense_169/bias/m
':%2Adam/dense_168/kernel/v
!:2Adam/dense_168/bias/v
':%2Adam/dense_169/kernel/v
!:2Adam/dense_169/bias/v
?2?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321420
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321308
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321403
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321294?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_64_layer_call_fn_1321446
/__inference_sequential_64_layer_call_fn_1321336
/__inference_sequential_64_layer_call_fn_1321433
/__inference_sequential_64_layer_call_fn_1321363?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_1321236?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_65?????????
?2?
F__inference_dense_168_layer_call_and_return_conditional_losses_1321457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_168_layer_call_fn_1321466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_169_layer_call_and_return_conditional_losses_1321476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_169_layer_call_fn_1321485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5B3
%__inference_signature_wrapper_1321386input_65?
"__inference__wrapped_model_1321236p	
1?.
'?$
"?
input_65?????????
? "5?2
0
	dense_169#? 
	dense_169??????????
F__inference_dense_168_layer_call_and_return_conditional_losses_1321457\	
/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ~
+__inference_dense_168_layer_call_fn_1321466O	
/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_dense_169_layer_call_and_return_conditional_losses_1321476\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ~
+__inference_dense_169_layer_call_fn_1321485O/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321294h	
9?6
/?,
"?
input_65?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321308h	
9?6
/?,
"?
input_65?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321403f	
7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_64_layer_call_and_return_conditional_losses_1321420f	
7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_64_layer_call_fn_1321336[	
9?6
/?,
"?
input_65?????????
p

 
? "???????????
/__inference_sequential_64_layer_call_fn_1321363[	
9?6
/?,
"?
input_65?????????
p 

 
? "???????????
/__inference_sequential_64_layer_call_fn_1321433Y	
7?4
-?*
 ?
inputs?????????
p

 
? "???????????
/__inference_sequential_64_layer_call_fn_1321446Y	
7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
%__inference_signature_wrapper_1321386|	
=?:
? 
3?0
.
input_65"?
input_65?????????"5?2
0
	dense_169#? 
	dense_169?????????