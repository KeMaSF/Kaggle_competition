



from difflib import SequenceMatcher

# some tutorial
# https://medium.com/@zhangkd5/a-tutorial-for-difflib-a-powerful-python-standard-library-to-compare-textual-sequences-096d52b4c843


a = """The cat is sleeping on the red sofa."""
b = """i am a great mean"""

seq_match = SequenceMatcher(None, a, b)
ratio = seq_match.ratio()
print(ratio)  # Check the similarity of the two strings

# The output similarity will be a decimal between 0 and 1, in our example it may output:
# the smaller, more difference



