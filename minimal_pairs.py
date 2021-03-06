import pickle

data = [#agent
['John', 'killed', 'the', 'deer', 'with', 'a', 'crossbow', 'while', 'out',
'hunting', 'in', 'the', 'woods', '.', 2,1,1],
#patient
['John', 'killed', 'the', 'deer', 'with', 'a', 'crossbow', 'while', 'out',
'hunting', 'in', 'the', 'woods', '.', 2,4,1],
['A', 'deer', 'was', 'killed', 'in', 'the', 'woods', 'by', 'a',
'hunter', 'with', 'a', 'crossbow', '.',4,2,1],
['A', 'deer', 'was', 'killed', 'in', 'the', 'woods', 'by', 'a',
'hunter', 'with', 'a', 'crossbow', '.', 4,10,1],
['John', 'kissed', "Mary", 'on', 'the', 'beach', '.', 2,1,1],
['John', 'kissed', "Mary", 'on', 'the', 'beach', '.', 2,3,1],
['Mary', 'was', 'kissed', 'by', 'John', 'on', 'the', 'beach', '.', 3,1,1],
['Mary', 'was', 'kissed', 'by', 'John', 'on', 'the', 'beach', '.', 3,5,1],
]

with open("minimal_pair.data", "wb") as filename:
  pickle.dump(data,filename)
