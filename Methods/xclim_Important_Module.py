"""""
xclim中重要的模塊

"""""
# todo Compare a dataArray to a threshold using given operator, return Boolean mask of the comparison
from xclim.indices.generic import compare
# todo Calculate the number of times some condition is met.
from xclim.indices.generic import count_occurrences,threshold_count
# Calculate the first time some condition is met.
from xclim.indices.generic import first_occurrence,last_occurrence
# Return a 0/1 mask when a condition is True or False.
from xclim.indices.generic import get_daily_events,spell_length,select_resample_op


