#import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from wordcloud import WordCloud, STOPWORDS


#import data
df = pd.read_csv("student-mat.csv", delimiter = ";")



#TUPLES example (and list technically)
age = df["age"].tolist()
final_grades = df['G3'].tolist()
age_grades = tuple(zip(age,final_grades))

#initialize plotting tuple
av1, av2, av3, av4, av5, av6 = 0,0,0,0,0,0
grade_list1 = []
grade_list2 = []
grade_list3 = []
grade_list4 = []
grade_list5 = []
grade_list6 = []


for i,j in age_grades:
    if (i==15):
        av1 = av1 + j; grade_list1.append(j)
    elif (i==16):
        av2 = av2 + j; grade_list2.append(j)
    elif (i==17):
        av3 = av3 + j; grade_list3.append(j)
    elif (i==18):
        av4 = av4 + j; grade_list4.append(j)
    elif (i==19):
        av5 = av5 + j; grade_list5.append(j)
    elif (i==20):
        av6 = av6 + j; grade_list6.append(j)

av1, av2, av3, av4, av5, av6 = av1/age.count(15),\
                                        av2/age.count(16),\
                                        av3/age.count(17),\
                                        av4/age.count(18),\
                                        av5/age.count(19),\
                                        av6/age.count(20)


std1, std2, std3, std4, std5, std6,std7,std8 = st.stdev(grade_list1), \
                                                st.stdev(grade_list2), \
                                                st.stdev(grade_list3), \
                                                st.stdev(grade_list4), \
                                                st.stdev(grade_list5), \
                                                st.stdev(grade_list6), \
                                                0,\
                                                0


plot1_tuple = ((15,av1),(16,av2),(17,av3),(18,av4),(19,av5),(20,av6))


#create a figure and axis
fig, ax = plt.subplots()
ax.bar(*zip(*plot1_tuple), yerr = [std1, std2, std3, std4, std5, std6], width = 0.5, color = ["midnightblue","darkslateblue","mediumpurple","pink","coral","darkorange"])

#set a title and lables
ax.set_title("Final Grades In Different Age Groups")
ax.set_xlabel('Student Age')
ax.set_xticklabels([av1,av2,av3,av4,av5,av6])
plt.xticks(np.arange(15,21, step=1), labels = ["15","16","17","18","19","20"])
ax.set_ylabel('Average Final Grade (Scored out of 20)')
for i,j in enumerate([av1,av2,av3,av4,av5,av6]):
    plt.text(i+15.01, j+0.25, str(round(j,2)))
plt.show()






#print(df[['absences','Dalc']])

#Dictionary Example

lit_factor = df["Dalc"].tolist()
num_absences = df["absences"].tolist()
ab1, ab2, ab3, ab4, ab5 = [], [], [], [], []

for i,j in enumerate(lit_factor):
    if(j == 1):
        ab1.append(num_absences[i])
    elif(j == 2):
        ab2.append(num_absences[i])
    elif(j == 3):
        ab3.append(num_absences[i])
    elif (j == 4):
        ab4.append(num_absences[i])
    elif (j == 5):
        ab5.append(num_absences[i])

#print(ab1)
#print(ab2)
#print(ab3)
#print(ab4)
#print(ab5)

plot2_dict = {1: ab1, 2: ab2, 3: ab3, 4: ab4, 5: ab5}

a = plot2_dict.get(1)
b = plot2_dict.get(2)
c = plot2_dict.get(3)
d = plot2_dict.get(4)
e = plot2_dict.get(5)

randomize_students = [i for i in range(0,len(a) + len(b) + len(c) + len(d) + len(e))]
randomize_students = np.random.permutation(randomize_students).tolist()
#print(randomize_students)

plt.clf()
plt.scatter(randomize_students[0:len(a)], a, color="blue", label = "Very Low Consumption")
plt.scatter(randomize_students[len(a): len(a)+len(b)], b, color = "green",  label = "Low Consumption")
plt.scatter(randomize_students[len(a)+len(b): len(a)+len(b)+len(c)], c, color='yellow', label = "Moderate Consumption")
plt.scatter(randomize_students[len(a)+len(b)+len(c): len(a)+len(b)+len(c)+len(d)], d, color='orange', label = "High Consumption")
plt.scatter(randomize_students[len(a)+len(b)+len(c)+len(d): len(a)+len(b)+len(c)+len(d)+len(e)], e, color='red', label = "Very High Consumption")
plt.xlabel("Student Number")
plt.ylabel("Number of Absences")
plt.title("Number of Absences for Students with Varying Levels of Workday Alcohol Consumption")
plt.legend
plt.legend()
plt.show()


# Read text
text = open('bible.txt').read()
stopwords = STOPWORDS


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")

    plt.show()



#generate wordcloud with my parameters
wordcloud = WordCloud(width = 3000,
                      height = 2000,
                      random_state=4,
                      background_color='black',
                      colormap='gist_ncar',
                      collocations=False,
                      stopwords = STOPWORDS)

wordcloud.generate(text)

#plot
plot_cloud(wordcloud)

#create file
#wordcloud.to_file('v1.png')










