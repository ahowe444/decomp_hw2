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

**Agent** - 2827/5669
**Patient** - 1906/5669
**Instrument** - 1499/ 5669
**Result** - 144/5669
** Destination** - 0/5669

Unfortunately, I wasn't able to develop a query that picked out any Destination roles from the data. 
This is probably a shortcoming of thematic role theory rather than proto-role theory because
it is not clear that there ought to be another role apart from the Location role. 
