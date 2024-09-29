import praw
import os

reddit = praw.Reddit(client_id='LvkNqRLxwft3OPWP0145hw',
                     client_secret='r81wYggBu4JhGsziY43JaGuOxjUBHQ',
                     user_agent="script by u/FracturedArc")

print(reddit.read_only)

# Specify the subreddit you want to scrape
subreddit_name = "Anxiety"
subreddit = reddit.subreddit(subreddit_name)

# Create the folder if it doesn't exist
folder_name = "disorder proj"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Create and open a text file to write the posts
file_path = os.path.join(folder_name, f"{subreddit_name}_posts.txt")
with open(file_path, 'w', encoding='utf-8') as file:
    top_posts = subreddit.top(limit=1000)
    
    saved_post_count = 0
    for post in top_posts:
        # Check if the post has content (selftext) or is a spoiler
        if post.selftext.strip() or post.spoiler:
            saved_post_count += 1
            post_content = f"\nPost {saved_post_count}: {post.title}\n"
            
            # Handle spoiler content
            if post.spoiler:
                post_content += "Content (Spoiler):\n"
            else:
                post_content += "Content:\n"
            
            post_content += f"{post.selftext}\n"
            
            # Write to file
            file.write(post_content)
            
            # Also print to console
            print(post_content)

print(f"{saved_post_count} Anxiety posts have been saved to {file_path}")