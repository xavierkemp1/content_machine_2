# content_machine_2
The purpose of this project is to automate the process of creating viral content, through the use of various apis. The project will follow the below flow:

1) Obtaining raw 'potential content'
   Top performing text content will be scraped from reddit and x using their apis (vauge on how to obtian/ determine the best potential posts, gpt help expand this). The X accounts and subreddits will be able to be set in a control pannel file. Keys and secrets will be held in a local .env file. Output of this stage will be a set of normalised text post objects.

2) Filtering
   posts will then be filtered for blacklist words, remove duplicates, and sorted into short, medium and long groups, based on the number of words in a post. after this, remaining posts will be sent to openai api and ranked in order of potential virality, based on the following criteria:
  hook strength
  emotional charge
  clarity
  relatability
  comment bait
  short-form suitability
Output from this stage would be the top x short, medium and long sets of filtered text post objects.

3) Improvements
   Successful posts would then be sent back to openai api to edit successful posts to maximise potential virality, returning:
  better hook
  rewritten, sligtly improved text for narration
  title
  caption
  hashtags
  output for this section will be a list of potentially viral posts, ready for tts

4) Production
   realistic tts for the content will be implimented. captions and background video will be added in 9:16 format, so the content can be viewed as a viral reel/ tiktok.
  (more detail)

5) Posts will then be saved as mp4 files in folders based on their content, ready for the user to post on chosen short form content platforms.
