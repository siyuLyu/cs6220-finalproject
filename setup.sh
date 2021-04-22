mkdir -p ~/.streamlit/
echo "[general]
email = \"lyu.si@husky.neu.edu\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml