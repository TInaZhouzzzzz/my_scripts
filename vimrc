set number
set hls
set ts=4
set expandtab
"set autoindent
set mouse=a


set tabstop=2 " tab键宽度为4
set softtabstop=2 " 统一缩减为4
set shiftwidth=2
set syntax=on "语法高亮
"代码补全
set completeopt=preview,menu 
"突出显示当前行
set cursorline
"映射全选+复制 ctr + a
map <C-A> ggVGY
map! <C-A> <Esc> ggVGY
map <F12> gg=G
" 选中状态下 ctr+c 复制
vmap<C-c> "+y


let mapleader=" "
nmap <leader>q :q<CR>
nmap <leader>s :w<CR>
nmap <leader>e :tabnew<Space>
nmap <leader>f :tabp<CR>
nmap <leader>g :tabn<CR>
nmap <leader>d :vert diffsplit<Space>
nmap <leader>v :vs<Space>
