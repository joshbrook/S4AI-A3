# W3-LinAlg-Neural Network
### Skills 4 AI Assignment 3 - Linear-Algebra-based Neural Networks
#### Group 9

## Coding tasks

- Find 3 papers related to this topic. For instance look for papers using the MNIST dataset. Create a short introduction referencing these papers. 
- Define three characters from your own name, and draw them in a 5x5 matrix. E.g.if you name is Sieuwert, make SIE, or SUT, … Be unique.
- Create four variations for each character. Make variation one of each character by redrawing it differently, e.g. emphasizing one of the strokes. Variant 2 should be based on slightly blurring the first version. Variant 3 should add some noise to a few pixels of the image. Variant four should combine noise and blurring. 
- Make a correlation matrix for your inputs (12 x 12 matrix) by taking inproducts between all input values. How similar are your inputs to each other? Explain from the correlation matrix which two characters are most likely to be confused. 
- Create a matrix NN1 to recognise your three characters. It should work on all variations, so perhaps use a combination/average version of the three characters. Make improvements to find best matrix. Try to make it so that the matrix gives an equally high output for each input character, to make comparison easy.
- Test your matrix on all your inputs and show the scores, for instance in bar chart. Evaluate the score: does the correct answer indeed get the highest score? Is the difference between the scores big enough to set a simple threshold value?
- Really test your network: make four or more inputs and use NN1 on it. Find multiple inputs (3 or more) that are not correctly classified. Check what happens if you input all 1's or all zeros? Try to make an incorrectly classified character by changing only one pixel. Is thsi possible? Can you do it by only changing two pixels? Three pixels?
- Find multiple inputs (not all zeros) so that NN1 cannot make a decision: it gives exactly equal values for all characters. Is there a linear algebra based method for finding such inputs? Describe how you can create such counterexamples. 

## Report requirements

- Title page with unique, relevant title like a research paper and your names
- An introduction where you explain what optical character recognition is and  cite three or more different scientific papers on optical character recognition
- A section "problem statement" where you show your three characters and four variations and explain the task. Explain how you created the variations and why these are relevant.
- A section "results" where you show the correlation matrix, and NN1 and you show using bar charts how the characters were recognized. Try to show a threshold value in the chart. Also show the other testing results. Explain how NN1 was created.
- A section "explanation" where you explain whether the accuracy in your view is good enough for actual use, you comment on how it could be improved further. Also comment on whether it was difficult to find inputs that were not correctly classified. Finally, comment on what thresholds would be if you could add a “Do not know” outcome.
- Include your source code as appendix
