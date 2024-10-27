PROMPT = {}

## eval docs
PROMPT['eval_doc_instr'] = '''
Instruction: You are given an query and an exerpt of a document. 
If the document contains relevant information output:
#relevant\n
If the document is not relevant output:
#irrelevant\n 
if the document is irrelevant do not provide a summary. 
Strictly follow this instructions.

Query: Who won the ncaa football national championship played in 2016?
Doc: The 2016 NCAA football national championship was won by the University of Alabama. In a thrilling game held on January 11, 2016, Alabama defeated Clemson University with a final score of 45-40. This victory marked Alabama’s 16th national title, solidifying their reputation as one of college football’s most successful programs.
'''

PROMPT['eval_doc_answ1'] = '''#relevant\n'''

PROMPT['eval_doc_ex2'] = '''Query: Who won the ncaa football national championship played in 2016?
Doc:  Jeju Island is significant due to its unique volcanic landscapes, including Hallasan Mountain and its lava tubes, which are UNESCO World Heritage sites.
'''

PROMPT['eval_doc_answ2'] = '''#irrelevant\n'''

PROMPT['eval_doc_ex3'] = '''Query: Why is Jeju Island significant?
Doc: Jeju Island holds great significance due to its unique volcanic landscapes, including the UNESCO World Heritage site, Hallasan Mountain, and its lava tubes. As the largest island in South Korea, it is a top tourist destination known for its natural beauty, traditional culture, and historical landmarks. Jeju is also an important ecological site, home to a diverse range of flora and fauna, making it both culturally and environmentally valuable.
'''

PROMPT['eval_doc_answ3'] = '''#relevant\n'''

# refine query
PROMPT['refine_query_instr'] = '''
    Instruction: You are given an query and an exerpt of a document. 
    Use contextual information to create just 10 new questions that are specific to the original query.
    Strictly follow this instructions.

    Original Query: Why is Jeju Island significant?
    Context information: ['Jeju Island is significant due to its unique volcanic landscapes, including Hallasan Mountain and its lava tubes, which are UNESCO World Heritage sites.',
    'It is the largest island in South Korea and is a popular destination for tourists seeking natural beauty and cultural experiences.',
    'The island is also home to a rich biodiversity, featuring various plant and animal species that make it an ecological treasure.',
    'Additionally, Jeju has cultural landmarks and traditions, which add to its historical and cultural importance.']
    '''

PROMPT['refine_query_answ1'] = '''
    ['### Why are Jeju Island’s volcanic landscapes considered significant?',
    '### How does Hallasan Mountain contribute to Jeju Island's significance?',
    '### What makes Jeju’s lava tubes noteworthy in terms of cultural or natural heritage?',
    '### How does Jeju Island's status as the largest island in South Korea add to its importance?',
    '### Why is Jeju Island a popular destination for tourists seeking natural beauty?',
    '### How does Jeju Island’s biodiversity make it an ecological treasure?',
    '### What types of unique plant and animal species are found on Jeju Island?',
    '### In what ways do Jeju Island’s cultural landmarks add to its historical importance?',
    '### How do Jeju’s traditions contribute to its cultural significance?',
    '### Why is Jeju Island’s UNESCO World Heritage status important to its overall significance?']
    '''

PROMPT['refine_query_ex2'] = '''
    Original Query: Who won the ncaa football national championship played in 2016?
    Context information: ['The NCAA football national championship game in 2016 was held on January 11 and featured a matchup between the University of Alabama and Clemson University.', 
    'In a close and thrilling game, the University of Alabama emerged victorious with a final score of 45-40.',
    'This win marked Alabama’s 16th national title, further solidifying its status as a dominant program in college football.',
    'The game was memorable for its intensity and high level of competition, captivating fans across the country.']
    '''

PROMPT['refine_query_answ2'] = '''
    ['### On what date was the 2016 NCAA football national championship game held?',
    '### Which two teams competed in the 2016 NCAA football national championship?',
    '### What was the final score of the 2016 NCAA football national championship game?',
    '### How did the University of Alabama’s victory impact its record in college football?',
    '### Why was the 2016 NCAA football national championship game considered thrilling?',
    '### How many national titles had Alabama won with this 2016 victory?',
    '### What made the 2016 championship game memorable for football fans?',
    '### How did Clemson perform against Alabama in the 2016 championship game?',
    '### What elements of the game highlighted the high level of competition?',
    '### In what ways did Alabama’s victory solidify its dominance in college football?']
    '''

# make new answer
PROMPT['new_answer_instr']='''
    Instruction: You are given an query and an exerpt of a document. 
    Answer the query using context information.
    Strictly follow this instructions.
    
    Query: what place did the New York Jets finish in 2015?
    Context information: ['In the 2015 NFL season, the New York Jets finished with a record of 10 wins and 6 losses.', 
    'Despite their strong record, the Jets narrowly missed the playoffs, finishing in second place in the AFC East division.', 
    'The team showed significant improvement from the previous season, with a notable performance by their defense and an improved offense.',
    'Although the Jets were close to securing a playoff spot, they ultimately fell short, marking the end of their 2015 season.']
'''

PROMPT['new_answer_answ1']='''
    The New York Jets finished in second place in the AFC East division in the 2015 NFL season.
'''

PROMPT['new_answer_ex2']='''
    Query: Who won the ncaa football national championship played in 2016?
    Context information: ['The 2016 NCAA football national championship game took place on January 11 and featured a competitive matchup between the University of Alabama and Clemson University.',
    'This victory marked Alabama's 16th national title, adding to their legacy as one of the most successful programs in college football history.',
    'The game was widely remembered for its high level of play and thrilling conclusion, captivating football fans across the nation.']
'''

PROMPT['new_answer_answ2']='''
    The University of Alabama won the NCAA football national championship played in 2016.
'''

PROMPT['final_answer_instr']='''
    Given the context information and not prior knowledge, 
    Generate an answer with multiple correct answers that satisfy an ambiguous question where the correct answer can vary depending on multiple interpretations
    Strictly follow this instructions.
    
    Query: Why is Jeju Island significant?
    Context information: ['Jeju Island is significant due to its unique volcanic landscapes, including Hallasan Mountain and its lava tubes, which are UNESCO World Heritage sites.',
    'It is the largest island in South Korea and is a popular destination for tourists seeking natural beauty and cultural experiences.',
    'The island is also home to a rich biodiversity, featuring various plant and animal species that make it an ecological treasure.',
    'Additionally, Jeju has cultural landmarks and traditions, which add to its historical and cultural importance.']
'''

PROMPT['final_answer_answ1']='''
    Jeju Island is significant for several reasons: its unique volcanic landscapes, including Hallasan Mountain and lava tubes, have earned it UNESCO World Heritage status. As the largest island in South Korea, it attracts tourists for its natural beauty and cultural experiences. The island is also an ecological treasure, home to a rich biodiversity of plant and animal species. Additionally, Jeju’s historical and cultural landmarks and traditions contribute to its cultural heritage, making it a place of both natural and cultural importance.
'''