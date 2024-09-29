import praw
import os

reddit = praw.Reddit(client_id='LvkNqRLxwft3OPWP0145hw',
                     client_secret='r81wYggBu4JhGsziY43JaGuOxjUBHQ',
                     user_agent="script by u/FracturedArc")

print(reddit.read_only)

subreddit_names = ["happy", "Positive", "CasualConversation"]

folder_name = "disorder proj"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, "normal_posts.txt")
with open(file_path, 'w', encoding='utf-8') as file:
    saved_post_count = 0
    
    for subreddit_name in subreddit_names:
        subreddit = reddit.subreddit(subreddit_name)
        top_posts = subreddit.top(limit=334)
        
        for post in top_posts:
            if post.selftext.strip() or post.spoiler:
                saved_post_count += 1
                post_content = f"\nPost {saved_post_count} from r/{subreddit_name}: {post.title}\n"
                
                if post.spoiler:
                    post_content += "Content (Spoiler):\n"
                else:
                    post_content += "Content:\n"
                
                post_content += f"{post.selftext}\n"
                
                file.write(post_content)

                print(post_content)

print(f"{saved_post_count} normal posts have been saved to {file_path}")
