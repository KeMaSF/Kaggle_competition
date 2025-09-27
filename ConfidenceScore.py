from lingua import Language, LanguageDetectorBuilder

# Build detector with all supported languages
detector = LanguageDetectorBuilder.from_all_languages().build()

# this one has score 1
# text = "Three healthcare providers: doctor, nurse, and technician.List these AI applications: diagnosis assistance, treatment planning, patient monitoring, drug interaction checking, medical imaging analysis, predictive health tracking, administrative automation, virtual nursing assistance, and personalized medicine recommendation."

# this one has score 0.5
text = "'NullPointerException', 'SegmentationFault', 'IndexOutOfBounds', 'SyntaxError', 'TypeError', 'MemoryLeak', 'InfiniteLoop', 'StackOverflow"



# Returns a list of LanguageConfidence objects
confidences = detector.compute_language_confidence_values(text)

# Find the English score
english_score = next(
    (c.value for c in confidences if c.language == Language.ENGLISH),
    0.0  # default if English is not in the list
)

print(f"English confidence: {english_score:.3f}")