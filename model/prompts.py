PROMPT = {}

## eval docs
PROMPT['eval_doc_instr'] = '''
You are given one or multiple queries and an exerpt of a document. If the document contains relevant information to answer the query output: #relevant\n 
If the document is not relevant output: #irrelevant\n.
Do not comment your output, but strictly follow these instructions.

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
You are given an ambiguous query and several excerpts of documents containing relevant information to answer this query. 
Given the query and the relevant documents, what kind of non-ambiguous follow up questions should be posed to answer the inital ambiguous query holisticaly?.
Please first consider why the initil query could be ambiguous and then pose curious follow-up question that dive deeper into an area not present in the provided documents. Do not comment the output.

Original Query: Apartheid ended in south africa during the presidency of?
Context Information: ['On 27 April 1994, a date later celebrated as Freedom Day, South Africa held its first elections under universal suffrage. The ANC won a resounding majority in the election and Mandela was elected president.',
'On 2–4 May 1990, the ANC met with the South African government at the Groote Schuur presidential residence in Cape Town, in what were touted as the first of several "talks about talks", intended to negotiate the terms for more substantive negotiations.',
'The apartheid system in South Africa was ended through a series of bilateral and multi-party negotiations between 1990 and 1993.']
'''
# Original Query: Why is Jeju Island significant?
# Context information: ['Jeju Island is significant due to its unique volcanic landscapes, including Hallasan Mountain and its lava tubes, which are UNESCO World Heritage sites.',
# 'It is the largest island in South Korea and is a popular destination for tourists seeking natural beauty and cultural experiences.',
# 'The island is also home to a rich biodiversity, featuring various plant and animal species that make it an ecological treasure.',
# 'Additionally, Jeju has cultural landmarks and traditions, which add to its historical and cultural importance.']
# '''

PROMPT['refine_query_answ1'] = '''['### Apartheid ended in South Africa at the end of whose South African presidency?,
### Apartheid ended in South Africa at the beginning of whose South African presidency?,
### Apartheid ended in South Africa during whose American presidency?
### When did Aparheid end in South Africa?
### What was Apartheid in South Africa?
### What Law ended Apartheid in South Africa?
### Which party did the president during which Apartheid ended in South Africa belong to?
]
'''

# PROMPT['refine_query_answ1'] = '''
# ['### Why are Jeju Island’s volcanic landscapes considered significant?',
# '### How does Hallasan Mountain contribute to Jeju Island's significance?',
# '### What makes Jeju’s lava tubes noteworthy in terms of cultural or natural heritage?',
# '### How does Jeju Island's status as the largest island in South Korea add to its importance?',
# '### Why is Jeju Island a popular destination for tourists seeking natural beauty?'
# '### How does Jeju Island’s biodiversity make it an ecological treasure?',
# '### What types of unique plant and animal species are found on Jeju Island?',
# '### In what ways do Jeju Island’s cultural landmarks add to its historical importance?',
# '### How do Jeju’s traditions contribute to its cultural significance?',
# '### Why is Jeju Island’s UNESCO World Heritage status important to its overall significance?']
# '''

PROMPT['refine_query_ex2'] = '''
Original Query: Who won the ncaa football national championship played in 2016?
Context information: ['The NCAA football national championship game in 2016 was held on January 11 and featured a matchup between the University of Alabama and Clemson University.', 
'In a close and thrilling game, the University of Alabama emerged victorious with a final score of 45-40.',
'This win marked Alabama’s 16th national title, further solidifying its status as a dominant program in college football.',
'The game was memorable for its intensity and high level of competition, captivating fans across the country.']
'''

PROMPT['refine_query_answ2'] = '''
['### Who won the 2016 season's ncaa football national championship?',
'### Who won the ncaa football national championship played in 2016?',
'### Who won the female ncaa football notional championship in 2016?',
'### Which teams participated in the finals of the ncaa football national championship in 2016?',
'### What was the score count at the end of the ncaa football national championship in 2016?
'''

# make new answer
PROMPT['new_answer_instr']='''
Instruction: You are given one or multiple queries and several exerpts of relevant documents to answer those queries. 
Please summarize the provided information so all relevant information regarding all the provided queries in represented in your answer.
Do not comment your answer and strictly follow this instructions.

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
You are provided an initial ambiguous query and several non-ambiguous follow-up queries each with context information to answer it. 
Generate an answer by considering all provided information, and answer the initial ambiguous question holisticaly.

Initial Query: Who won the 2016 ncaa football national championship?
context: The 13–1 Alabama Crimson Tide won the game, holding off the undefeated Clemson Tigers 45–40 in the fourth quarter. It was played at University of Phoenix Stadium in Glendale, Arizona on January 11, 2016, and was the culminating game of the 2015–16 bowl season.

Follow-up Query1: ### Who won the 2016 season's ncaa football national championship?
Context: The 13-1 Alabama Crimson Tide won the game, holding off the undefeated Clemson Tigers 45–40 in the fourth quarter. Accompanied by a talented receiving corps, Clemson's Heisman Finalist quarterback Deshaun Watson had a historic performance, setting the record for most total yards in national championship game history, with 478 yards (405 passing / 73 rushing) against the nation's third-ranked defense in Alabama, breaking the record previously set by Vince Young in the 2006 Rose Bowl. Following the game, the AP Poll also named Alabama as its top team of the season, giving Alabama their fourth title in seven seasons. Both Clemson and Alabama finished the season 14–1.

Follow-up Query2: ### Who won the ncaa football national championship played in 2016?
Context: The game was played between the winners of two pre-designated bowl games played on December 31, 2016: the Clemson Tigers, who defeated the Ohio State Buckeyes in the Fiesta Bowl, and the Alabama Crimson Tide, who defeated the Washington Huskies in the Peach Bowl. Having met in the previous year's championship game, the resulting title game between Clemson and Alabama became college football's first rematch between #1 and #2 in national championship game history.

''' 
#     Query: Why is Jeju Island significant?
#     Context information: ['Jeju Island is significant due to its unique volcanic landscapes, including Hallasan Mountain and its lava tubes, which are UNESCO World Heritage sites.',
#     'It is the largest island in South Korea and is a popular destination for tourists seeking natural beauty and cultural experiences.',
#     'The island is also home to a rich biodiversity, featuring various plant and animal species that make it an ecological treasure.',
#     'Additionally, Jeju has cultural landmarks and traditions, which add to its historical and cultural importance.']
# '''

PROMPT['final_answer_answ1']='''
The 2015 - 2016 season's ncaa national football championship game was played between the Clemson Tigers and the Alabama Crimson Tide on January 11, 2016. The Alabama Crimson Tide won the game by holding off the undefeated Clemson Tigers 45–40 in the fourth quarter.
'''
# PROMPT['final_answer_answ1']='''
#     Jeju Island is significant for several reasons: its unique volcanic landscapes, including Hallasan Mountain and lava tubes, have earned it UNESCO World Heritage status. As the largest island in South Korea, it attracts tourists for its natural beauty and cultural experiences. The island is also an ecological treasure, home to a rich biodiversity of plant and animal species. Additionally, Jeju’s historical and cultural landmarks and traditions contribute to its cultural heritage, making it a place of both natural and cultural importance.
# '''