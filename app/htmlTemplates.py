
# css = """
# <style>
#     .chat-container {
#         max-width: 700px;
#         margin: 0 auto;
#         padding: 20px;
#         background-color: #ffffff;
#         border-radius: 10px;
#         border: 2px solid #e0e0e0; /* Outer border for the entire conversation */
#         box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#         overflow-y: auto; /* Enable scrolling if the content exceeds container height */
#         max-height: 600px; /* Set maximum height */
#     }
#     .message {
#         display: flex;
#         align-items: flex-start;
#         margin-bottom: 15px;
#         border-radius: 10px;
#         padding: 10px;
#         background-color: #fefefe;
#     }
#     .message.bot .message-content {
#         background-color: #e6f7ff;
#         color: #005f99;
#         border: 1px solid #005f99;
#         border-radius: 10px;
#         padding: 10px;
#     }
#     .message.user .message-content {
#         background-color: #d1ffd1;
#         color: #006400;
#         border: 1px solid #006400;
#         border-radius: 10px;
#         padding: 10px;
#     .avatar {
#         width: 40px;
#         height: 40px;
#         border-radius: 50%;
#         background-size: cover;
#         margin-right: 10px;
#     }
#     .avatar.bot {
#         background-image: url('https://img.icons8.com/ios/50/message-bot.png');
#         background-size: cover;
#         filter: invert(100%);
#     }
#     .avatar.user {
#         background-image: url('https://img.icons8.com/puffy/32/user.png');
#         background-size: cover;
#         filter: invert(100%);
#     }
# </style>
# """

# bot_template = """
# <div class="message bot">
#     <div class="avatar bot"></div>
#     <div class="message-content">
#         {{MSG}}
#     </div>
# </div>
# """

# user_template = """
# <div class="message user">
#     <div class="avatar user"></div>
#     <div class="message-content">
#        {{MSG}}
#     </div>
# </div>
# """
css = """
<style>
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    .message.bot {
        justify-content: flex-start;
    }
    .message.user {
        justify-content: flex-end;
    }
    .message-content {
        max-width: 70%;
        padding: 12px 15px;
        border-radius: 12px;
        font-size: 16px;
        line-height: 1.5;
    }
    .message.bot .message-content {
        background-color: #212d33;
        color: #60b3e6;
        border: 1px solid #60b3e6;
    }
    .message.user .message-content {
        background-color: #212d33;
        color: #5ee05e;
        border: 1px solid #5ee05e;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-size: cover;
        margin-right: 10px;
    }
    .avatar.bot {
        background-image: url('https://img.icons8.com/ios/50/message-bot.png');
        background-size: cover;
        filter: invert(100%);
    }
    .avatar.user {
        background-image: url('https://img.icons8.com/puffy/32/user.png');
        background-size: cover;
        filter: invert(100%);
    } 
</style>
"""

bot_template = """
<div class="message bot">
    <div class="avatar bot"></div>
    <div class="message-content">
        {{MSG}}
    </div>
</div>
"""

user_template = """
<div class="message user">
    <div class="avatar user"></div>
    <div class="message-content">
        {{MSG}}
    </div>
</div>
"""
