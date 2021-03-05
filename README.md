# Will Howe
# Decomp HW 2
---
## 3.1 Data Collection
---
For my queries I used the docker container for all my querying inside of a Jupyter Notebook.
The jupyter notebook is named `decomp_hw2.ipynb`. Inside of the Jupyter Notebook, I turn my
queries into 2D arrays in Python and then pickle them. This makes it fairly easy to use the data
in my data_reader of allennlp. 

### Q1

My roles are: Agent, Patient, Result, Instrument, and Destination. I chose Patient because it
seemed like a logical equivalent to Agent and it would only involve the reversal of a couple
proto-role features. It also had a similar frequency in the data to *Agent*.

Then, I mostly chose the other three because they occur with lower frequency in the data,
and I wanted to see if I could develop their respective proto-role definitions. It is a bit
hard to tell whether low model performance on these low-frequency roles is due to a bad proto-
role definition or due to insufficient training data. 

### Q2

**Patient** - As mentioned above, to motivate my definition for Patient, I simply took the 
*volition* and *instigation* roles which are characteristic of the Agent definition and reversed
them to indicate that Patient are used acted upon in a event instead of being the actor. 

**Instrument** - For this role, I considered sentences like, *The gladiator killed the lion with a sword.* where the instrument here is *the sword*. From here I realized that instruments are probably not
sentient, probably not aware, and definitely do get used and are existent during the event.

**Result** - Results are tricker to define, but in a a sentence like, *John hammered the metal flat*,
there is definitely a change of state. There is also a sense that the result doesn't exist before
the event predicate, but does exist after. So I use the features `existed_before`, `existed_after`, and 
`change_of_state`. 

**Destination** - This role is also hard to define. Here I note that a destination argument crucially
involves a change of location, otherwise the argument should simply be a Location role. So I use the 
features `location`, `change_of_location`, and `existed_before`. I assume that locations in the physical
world necessarily exist before someone can travel to a place as a *Destination*.


### Q3

**Agent** - 2827/5669 <br>
**Patient** - 1906/5669 <br>
**Instrument** - 1499/ 5669 <br>
**Result** - 144/5669 <br>
**Destination** - 0/5669 <br>

Unfortunately, I wasn't able to develop a query that picked out any Destination roles from the data. 
This is probably a shortcoming of thematic role theory rather than proto-role theory because
it is not clear that there ought to be another role apart from the Location role. 

Some roles occur more frequently than other just as Dowty predicted, there are a small set of frequently
occuring roles and there is also a long tail of less common roles. It is harder to develop proto-role
definitions for these and harder to collect a lot of data on them. 

## 3.2 Modeling

### Q1 
My model architecture basically follows pretty cleanly from Joe Barrow's tutorial as
well as from the proto-role prediction paper. The task in this homework (semantic role labeling from 
proto-roles) is essentially the reverse of the task in the Rudinger SPRL paper. So, accordingly, it probably makes sense to borrow ideas from that architecture as well.

1) First I embed my tokens into GloVe embeddings. 
2) Next, each embedding is passed through a Bidirectional LSTM sequence encoder into an output of
[num_batch, num_sequence, embedding_dim]
3) I extract the LSTM hidden states for the predicate and argument positions in the sentence as shown in 
the Rudinger paper.
4) I concatenate these 50 dim vectors into a 100 dim vector and pass it through a feed forward linear
layer. There is no ReLU since it will only reduce information content. 
5) Feed forward output head straight to the CrossEntropyLoss which itself computes the softmax. 

### Q2
I did not use any additional features, because I figured that using a BiLSTM which is a recurrent
architecture would automatically encode information more effectively than
I could have by feature engineering. 


**NOTE** I tried to add other linear layers and ReLU between them but this reduced accuracy.

## 3.3 Exploration

### Q1
There are in fact examples that seem to be wrong. Just take this sentence from my Agent train data for example: *Over 300 Iraqis are reported dead and 500 wounded in Fallujah alone.* The predicate here is labeled 5 which refers to *reported* and the argument is 3, *Iraqis*, but here *Iraqis* is not agentive, instead it is closer to an Experiencer/Theme situation. There of course are certain modification that could be made to the proto-role feature defintion such as further restricting the values rather than 
using a binary > 0, < 0 dichotomy. But, this strategy risks cutting out *valid* agents that didn't meet the more restrictive critera. We probably have to accept that there will be noise in our data since we 
can never craft a perfect rule, especially since our data is derived from non-linguist crowdsourced annotators. 

