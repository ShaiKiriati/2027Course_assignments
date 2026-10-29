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

- Result: makes basic version (not here)

2. is there an alternative to SAM that'll work better? that's just the one I know

- Result: recommends SAM3

3. update to SAM3

- Result: makes version with SAM3, explains how to install the package
  
4. add to concept list objects you may find in a closet, a dresser, a refrigerator, a dishwasher, a pantry or a suitcase
   
- Result: adds objects to list
  
5. now let's add the second step: make an interactive display of the image with the list of chosen object names printed below. 
  The interactive part is that when you left-click your mouse over a pixel belonging to one of the objects in the list,
  the object name in the list changes color and an outline of the clicked object is displayed over the image

- Result: makes SAM3 2nd version
  
6. (written while waiting for access for SAM3) write an alternative version using SAM 2
   
- Result: makes SAM2 version here

7. continuing with the SAM3 version, let's improve the GUI to be more like a classic hidden object game:
  when an object is chosen, it is stricken off the list. When the next object is chosen, the previous ones stay with strike
  when an object is chosen, rather than being marked by a red outline, it gets a dark overlay. The overlay stays when you click more objects

- Result: makes next SAM3 version. Suggests making a "new game" key.

8. add this "new game" key

- Result: adds the option to click N for a new game

9. add concept groups:
  desk
  toys
  games
  bathroom
  bedroom

- Result: adds more concept keywords in categories

10. if I don't want the user to input the group, would it be faster to just run for everything, or use image recognition to describe the image and extract the relevant groups from that?

- Result: recommends image recognition (as expected)

11. implement this modification

- Result: "segment_and_name_sam3.py now picks the concept groups itself, and this is the default (--groups auto)" adds N groups to the search (default: 3) besides "general"

12. (after I discovered my computer doesn't have an nvidia GPU and got suggestions for speedup) implement the changes

- Result: implemented changes: "--fast

  It runs SAM 3 at 560 pixels instead of 1008, following the approach in the Transformers docs. That should cut CPU time a lot, at some cost in accuracy.
  Result caching

  How it works: after SAM 3 finds the objects, the script saves them, and the next run on the same photo loads them instantly. It skips both SAM 3 and the scene recognition."

13. Would it speed up the process while keeping randomness to go over the list of concepts to search randomly rather than by order and just stop as soon as enough objects for the game have been found?

- Result: "What I’d implement

  Shuffle the concept list, optionally with a --seed for reproducibility.
  Process the concepts one by one and de-duplicate every 10 or so.
  Stop once the pool reaches --pool N (default around 30), or when the list runs out.
  Cap instances per concept.
  Include the pool size in the cache key."

14. implement these changes

- Result: "Options
  --pool N: change the stopping point. It’s raised to 20 if you pick less, since a game uses up to 20 objects. --pool 0 searches every concept, as before.
  --seed N: makes the concept order reproducible, which helps when comparing runs. Without it, each run shuffles differently.
  Cache: the pool size, seed and per-concept cap are part of the cache key. If you change any of them, the photo is searched again rather than loading a stale result."
