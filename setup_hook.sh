#!/bin/sh

cat <<'EOF' > .git/hooks/post-checkout
#!/bin/sh
find . -name "*.sh" -exec chmod +x {} \;
EOF
chmod +x .git/hooks/post-checkout

