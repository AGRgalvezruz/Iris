
# PonyGE2: grammatical evolution

### Grammar file 


```python
<e> ::= (<e> <lo> <e>) | <var> | <c>
<lo> ::= < | > | >= | <=
<var> ::= x[<varidx>]
<varidx> ::= GE_RANGE:dataset_n_vars
<c> ::= <d>.<d> | -<d>.<d>
<d> ::= GE_RANGE:8
```

### Iris-setosa vs. Iris-virginica Iris-versicolor


```python
Best:
  Fitness:	 1.0
  Phenotype: (x[1] > x[2])
  Genome: [10758, 53265, 19765, 85712, 90496, 52502, 30249, 44806, 63468, 21346, 89927, 19408]

  ave_fitness : 	 0.7912497488900199
  ave_genome_length : 	 211.216
  ave_tree_depth : 	 9.21920668058455
  ave_tree_nodes : 	 60.41544885177453
  ave_used_codons : 	 41.90187891440501
  best_fitness : 	 1.0
  gen : 	 50
  invalids : 	 1421
  max_genome_length : 	 488
  max_tree_depth : 	 17.0
  max_tree_nodes : 	 310.0
  max_used_codons : 	 214.0
  min_genome_length : 	 12
  min_tree_depth : 	 4.0
  min_tree_nodes : 	 4.0
  min_used_codons : 	 3.0
  runtime_error : 	 0
  time_adjust : 	 0
  time_taken : 	 0.2995169162750244
  total_inds : 	 25500
  total_time : 	 17.54463815689087
  unique_inds : 	 16405
  unused_search : 	 35.66666666666667

```

### Iris-virginica vs. Iris-versicolor


```python
Best:
  Fitness:	 0.9393815708101423
  Phenotype: (1.6 >= x[3])
  Genome: [95055, 61373, 2864, 82353, 95070, 95702, 82720, 91738, 68655, 65445, 15489, 23757, 77124, 4885, 7545, 87206, 53793, 55675, 64053, 68789, 97883, 13267, 80909, 18534, 47876, 50749, 14230, 38554, 91422, 26864, 20071, 75678, 27853, 72783, 80998, 46583, 62797, 32203, 8371, 319, 8144, 84526, 10254, 98261, 32377, 20578, 27434, 20990, 47112, 98426, 24275, 70635, 92249, 75890, 53107, 53304, 31724, 12380, 57394, 8570, 36336, 29213, 16606, 21646, 97420, 69263, 39915, 36622, 38704, 20056, 36104, 91905, 52079, 41338, 63912, 38732, 14161, 86892, 18368, 95865, 35767, 56418, 56189, 85054, 65882, 10498, 55860, 40913, 48623, 67769, 66409, 47760, 1154, 98225, 74827, 33664, 67291, 68138, 70654, 31333, 53775, 72517, 7276, 6284, 76057, 83438, 95853, 6508, 57935, 98626, 96703, 1148, 169, 73693, 25275, 14494, 1044, 12734, 18824, 32941, 25556, 25688, 49065, 17638, 76646, 22061, 7097, 42766, 45916, 5531, 36724, 48088, 80693, 75268, 71286, 576, 36221, 86720, 63714, 68317, 55926, 8480, 48653, 81472, 66200, 36307, 8076, 96337, 68526, 18319, 23678, 43027, 41432, 32416, 18249, 24914, 71853, 86819, 25856, 45346, 63624, 79065, 85969, 25698, 63105, 26150, 65478, 95762, 39639, 82479, 63347, 21961, 98242, 36818, 68000, 92486, 88459, 23581, 4416, 43192, 38492, 79458, 58396, 60695, 62507, 87468, 54346, 92300, 87621, 38715, 67550, 45638, 31818, 66962, 68159, 70405, 8399, 79422, 10107, 24054, 10242, 83150, 8409, 5460, 82786, 29699, 20483]

  ave_fitness : 	 0.4334430737209122
  ave_genome_length : 	 274.804
  ave_tree_depth : 	 9.719262295081966
  ave_tree_nodes : 	 66.22950819672131
  ave_used_codons : 	 45.807377049180324
  best_fitness : 	 0.9393815708101423
  gen : 	 50
  invalids : 	 719
  max_genome_length : 	 500
  max_tree_depth : 	 17.0
  max_tree_nodes : 	 252.0
  max_used_codons : 	 175.0
  min_genome_length : 	 17
  min_tree_depth : 	 4.0
  min_tree_nodes : 	 4.0
  min_used_codons : 	 3.0
  runtime_error : 	 0
  time_adjust : 	 0
  time_taken : 	 0.2940962314605713
  total_inds : 	 25500
  total_time : 	 17.273632526397705
  unique_inds : 	 17299
  unused_search : 	 32.160784313725486

```
