PROMPT = {}

PROMPT['eval_doc_instr'] = '''
You are given an query and an exerpt of a document. If the document contains relevant information to answer the query output: #relevant\n 
If the document is not relevant output: #irrelevant\n.
Do not comment your output, but strictly follow these instructions.

Query: Who has the highest goals in world football?
Doc: in top - level association football competitions, 25 players have scored 500 or more goals in both club and international football, according to research by the iffhs, [ 1 ] first published in 2007. [ 2 ] taking into account competitions of all levels, 77 players have reached the milestone, according to research by the rsssf, [ 3 ] an organisation described by german newspaper der spiegel as a " wikipedia of football statistics ". [ 4 ] hungarian imre schlosser was the first to reach the 500 - goal mark, doing so in 1927 shortly before his retirement. [ 5 ] nine players have accomplished the feat at a single club : josef bican ( slavia prague ), jimmy jones ( glenavon ), jimmy mcgrory ( celtic ), joe bambrick ( linfield ), lionel messi ( barcelona ), gerd muller ( bayern munich ), pele ( santos ), fernando peyroteo ( sporting cp ), and uwe seeler ( hamburg ). [ 6 ] of these nine, messi scored the most, with 672 goals between his debut in 2004 and his departure in 2021. [ 7 ] fifa, the international governing body of football,
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
''' #if the document is irrelevant do not provide a summary. Strictly follow this instructions.; “summary of the important information of the document"

PROMPT['eval_doc_answ1'] = '''#relevant\n''' #25 players scored 500 goals or more in both club and international football. Nine players achieved this feat at a single club, with Lionel Messi scoring the most with 672 goals.'''

PROMPT['eval_doc_ex2'] = '''Query: Who has the highest goals in world football?
Doc:  # table : sounds of silence studio album by simon & garfunkel releasedjanuary 17, 1966 ( 1966 - 01 - 17 ) recorded ; april 5 – december 22, 1965 ; ( 
'''

PROMPT['eval_doc_answ2'] = '''#irrelevant\n'''

PROMPT['eval_doc_ex3'] = '''Query: Who has the highest goals in world football?
Doc:  matches considered official internationals by the opposing sides, which would make him the first footballer to score 50 or more international goals, ahead of imre schlosser, and was the fastest to achieve the feat, scoring his 50th goal in his 32nd official international match, with a four - goal haul against hungary on 31 may 1909. [ 17 ] puskas overall scored 84 goals in his international career, [ 11 ] and remained the highest international goalscorer for 24 years following his 84th goal in 1956 against austria, until mokhtar dahari of malaysia broke the record in the merdeka tournament after scoring his 85th goal on 27 october 1980 against kuwait and he went on to score 89 goals for his country in 142 international appearances. [ 18 ] [ 19 ] [ 20 ] in 2004, ali daei of iran broke the record after scoring his 90th goal against lebanon. [ 21 ] [ 22 ] daei also became the first player to score over 100 goals in international football, ending his career with 108 in total. [ 23 ] [ 24 ] his 100th goal came on 17 november 2004, when he scored a four - goal haul against laos in a 2006 fifa world cup qualification match. [ 25 ] however,
'''

PROMPT['eval_doc_answ3'] = '''#relevant\n''' #Imre Schlosser, Ferenc Puskas, Mokhtar Dahari, Ali Daei, who all hold the record for most international goals. \n Ferenc Puskas was the first footballer to score 50 or more international goals. \n Ali Daei became the first player to score over 100 goals in international football.'''

PROMPT['refine_query_instr'] = '''
    Instruction: You are given an query and several excerpts of documents containing relevant information. 
    Given the query and the relevant documents, what kind of follow up questions should be posed to answer the inital query in the better way?.

    Original Query: Who has the highest goals in world football?
    Context information: '25 players scored 500 goals or more in both club and international football. Nine players achieved this feat at a single club, with Lionel Messi scoring the most with 672 goals.',
    'Cristiano Ronaldo holds the all-time record with 133 international goals.',
    'The document contains information about the highest goals in world football, but it does not provide a direct answer to the question. However, it mentions that Just Fontaine holds the record for most consecutive matches scored with 6 goals.',
    'Ferenc Puskas, who scored 84 goals in his international career and remained the highest international goalscorer for 24 years. He was later surpassed by Mokhtar Dahari of Malaysia, who scored 89 goals, and then Ali Daei of Iran, who broke the record with 108 goals.'
    '''

PROMPT['refine_query_answ1'] = '''
    '### Who are the 25 players who have scored 500 goals or more in world football and how many goals did Lionel Messi score for a single club?',
    '### Who currently holds the record for the most international goals in world football?',
    '### Who holds the record for most consecutive matches scored with goals in international football?',
    '### Who broke the record of Ferenc Puskas and became the new highest international goalscorer and how many goals did he score?'
    '''


PROMPT['new_answer_instr']='''
    Instruction: You are given an query and an exerpt of a document. 
    Make a sentence of answer with original query using context information.
    Strictly follow this instructions.
    
    Query: Who currently holds the record for the most international goals in world football?'
    Context information: ['25 players scored 500 goals or more in both club and international football. Nine players achieved this feat at a single club, with Lionel Messi scoring the most with 672 goals.',
    'Cristiano Ronaldo holds the all-time record with 133 international goals.',
    'The document contains information about the highest goals in world football, but it does not provide a direct answer to the question. However, it mentions that Just Fontaine holds the record for most consecutive matches scored with 6 goals.',
    'Ferenc Puskas, who scored 84 goals in his international career and remained the highest international goalscorer for 24 years. He was later surpassed by Mokhtar Dahari of Malaysia, who scored 89 goals, and then Ali Daiei of Iran, who broke the record with 108 goals.']
'''

PROMPT['new_answer_answ1']='''
    Cristiano Ronaldo currently holds the record for the most international goals in world football, with 133 goals.
'''