# Project: hidden objects real life
The project is an idea I've had for a while: A hidden object game, which is generated automatically using image segmentation tools from user-uploaded images.
The idea is to gamify the real-life task of finding specific objects in a messy scene.
This is based on the genre of hidden-object games, in which you get a scene packed with objects and need to find specific ones from a list you are given.
A simple familiar example: "Where's Waldo?", in which the objects are all under the group of people and the list of objects to locate only includes Waldo.

# list of prompts to Claude:

1. Make the following python program:
  user opens an image (png or jpeg)
  The program uses SAM or something similar to segment objects in the image
  The program then uses a machine vision/ language model to name the segmented objects.
  The program then chooses 5-20 objects randomly to create a list.
  The program displays the input image, with the chosen list of object names below
Result: makes basic version (not here)
2. is there an alternative to SAM that'll work better? that's just the one I know
Result: recommends SAM3
3. update to SAM3
Result: makes version with SAM3, explains how to install the package
4. add to concept list objects you may find in a closet, a dresser, a refrigerator, a dishwasher, a pantry or a suitcase
Result: adds objects to list
5. now let's add the second step: make an interactive display of the image with the list of chosen object names printed below.
  The interactive part is that when you left-click your mouse over a pixel belonging to one of the objects in the list,
  the object name in the list changes color and an outline of the clicked object is displayed over the image
Result: makes SAM3 version here
6. (written while waiting for access for SAM3) write an alternative version using SAM 2
Result: makes SAM2 version here
