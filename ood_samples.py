# Hand-crafted samples fed to the model
samples = [
    # very ambigious
    ("Fed raises interest rates percent", 1), # interest and rates : (noun, verb)
    ("Flies like a flower", 1), # flies and flower : (noun, verb)
    ("The can you left is rusted", 1), # can and left : (noun, verb)
    ("Time flies like a fish", 1), # flies and arrow : (noun, verb)
    ("The robber did fire his gun and just left", 1), # fire and left : (noun, verb) 
    ("Did she fish a fish", 1), # fish : (verb, noun)
    ("Fish attracts many flies", 1), # flies and fish : (noun, verb)
    ("As opposed to the Human race flies prefer to be dirty", 1), # flies and race : (noun, verb)
    # ambigious
    ("The boy took a left after the car", 1), # left : (noun, verb)
    ("Has the professor left yet", 1), # left : (noun, verb)
    ("Did he fire the gun", 1), # fire : (noun, verb)
    ("The fire truck is red", 1), # fire : (noun, verb)
    ("Do you like fish", 1), # fish : (verb, noun)
    ("Please mind the gap while exiting the train", 1), # train : (verb, noun)
    ("He will race the car", 1), # race : (verb, noun)
    ("When does the race start", 1), # race : (verb, noun)
    ("I really enjoyed the play", 1), # play : (verb, noun)
    ("Do you project to live in that city", 1), # project : (verb, noun)
    ("I am in a band and I play the guitar", 1), # play : (verb, noun)
    # expected not to be ambiguious
    ("He had coffee that morning", 0),
    ("That is Bob from Chicago", 0),
    ("He got up early to go to the forest", 0),
    ("She likes to read magazines", 0),
    ("The car was big green and nice", 0),
    ("The weather was nice so he decided to go out", 0),
    ("What do you prefer rice or pasta", 0),
    ("He prefered to go to the cinema rather than the theater", 0),
    ("She really wanted to learn German", 0), 
    ("The summer was very warm and humid", 0),
    ("Do you prefer summer or winter", 0)
]
