# NLP-Detective
# Part 1: Get Set Up
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')  # POS tagging
nltk.download('maxent_ne_chunker')           # Named entity recognition
nltk.download('words')                       # Word lists for NE chunker

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Part 2: Create a Mystery Story
story = """Detective John arrived in Paris to investigate a robbery.
The FBI asked him to work with the local police.
He met with Sarah at the Eiffel Tower.
They believed the stolen painting was hidden in the Louvre."""

# Part 3: Tokenize and POS Tag
tokens = word_tokenize(story)
tagged = pos_tag(tokens)

print("====== DETECTIVE CASE FILE ======\n")
print("Original Mystery Story:")
print(story)
print("\n--- Part-of-Speech Tagged Words ---")
print(tagged)

# Part 4: Named Entity Recognition
ner_tree = ne_chunk(tagged)

print("\n--- Named Entities Found ---")
print(ner_tree)

# Extract Named Entities: People, Locations, Organizations
people = []
locations = []
organizations = []

for subtree in ner_tree:
    if isinstance(subtree, Tree):
        entity = " ".join(token for token, pos in subtree.leaves())
        label = subtree.label()
        if label == "PERSON":
            people.append(entity)
        elif label == "GPE":  # GPE = Geo-Political Entity (locations)
            locations.append(entity)
        elif label == "ORGANIZATION":
            organizations.append(entity)

# Part 5: Analyze and Report
print("\n====== CASE ANALYSIS REPORT ======")
print("People Identified:", people)
print("Locations Identified:", locations)
print("Organizations Identified:", organizations)

# Real-World Application Reflection
print("\n====== REFLECTION ======")
print("How could this kind of NLP be used in the real world?")
print("""NLP techniques like POS tagging and Named Entity Recognition (NER) are crucial in various real-world applications.
They help in tasks like automated news summarization, extracting key information from documents,
chatbots understanding user queries, and even detecting fake news or identifying important subjects in legal or medical records.
For example, a law firm could use NER to quickly extract names of clients, dates, and case details from hundreds of legal files.""")
