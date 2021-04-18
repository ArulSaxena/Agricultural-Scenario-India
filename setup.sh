mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"arulsaxena9596@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
© 2021 GitHub, Inc.