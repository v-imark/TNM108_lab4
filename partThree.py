text = """This course deals with procedural generation of assets, primarily with the focus on assets for computer 
games and computer animations but not limited to these. 

The course was created by Stefan Gustavson and has been given by him 16 times until 2019. Unfortunately, 
he is plagued by a long-time illness that caused the course to be canceled 2020. For 2021, we could still not promise 
Stefan to be in shape to be able to give the course, but this time I volunteered to step in and handle the course. 
Stefan was still involved, as my advisor. He may or may not take active part in the course. (PS: He very much did, 
with half of the last lecture. Much appreciated!) This made the 2021 course a challenge for me, and to some extent 
for the participants, since the course had to be given with pretty extensive preparations in limited time. 

Fortunately, the 2021 course went very well and I hope I can keep up the quality. For 2022, much of the material is 
ready and I don’t plan any extensive changes. I plan to give the lectures on location this time. If that turns out to 
be too hard I might go back to on-line but at this time the plan is to be on location. 

When Stefan gets better, I have every intention of handing the course back to him, which means that I should limit my 
time investments. However, I have no plan in letting this make the course bad. I take on the course since I like the 
subject and have done work in the area before. My own courses (TSBK07 and TSBK03) border to it, for a reason. 

Stefan is giving me full freedom to handle the course my way. I will use Stefan's material a lot (of course I will!) 
but also model much after how I give my own courses. 

As you may have noted, this course page shares its looks and structure with my own courses but absolutely no other 
webpages that I know of. This is intentional: When you are on the course pages, it is obvious from the looks! This 
one is green, TSBK07 is blue and TSBK03 is red/black. And my demo/packages pages look totally different from these 
because they are not course pages (but resources for the course pages). I like it that way! """


from summa.summarizer import summarize
# Define length of the summary as a proportion of the text
#print(summarize(text, ratio=0.2))

print(summarize(text, words=50))

from summa import keywords
print("Keywords:\n",keywords.keywords(text))
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text,words=3))