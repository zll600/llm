def read_file_content(cls, file_path: str):
    # 根据文件扩展名选择读取方法
    if file_path.endswith(".pdf"):
        return cls.read_pdf(file_path)
    elif file_path.endswith(".md"):
        return cls.read_markdown(file_path)
    elif file_path.endswith(".txt"):
        return cls.read_text(file_path)
    else:
        raise ValueError("Unsupported file type")


def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
    chunk_text = []

    curr_len = 0
    curr_chunk = ""

    token_len = max_token_len - cover_content
    lines = text.splitlines()  # 假设以换行符分割文本为行

    for line in lines:
        # 保留空格，只移除行首行尾空格
        line = line.strip()
        line_len = len(enc.encode(line))

        if line_len > max_token_len:
            # 如果单行长度就超过限制，则将其分割成多个块
            # 先保存当前块（如果有内容）
            if curr_chunk:
                chunk_text.append(curr_chunk)
                curr_chunk = ""
                curr_len = 0

            # 将长行按token长度分割
            line_tokens = enc.encode(line)
            num_chunks = (len(line_tokens) + token_len - 1) // token_len

            for i in range(num_chunks):
                start_token = i * token_len
                end_token = min(start_token + token_len, len(line_tokens))

                # 解码token片段回文本
                chunk_tokens = line_tokens[start_token:end_token]
                chunk_part = enc.decode(chunk_tokens)

                # 添加覆盖内容（除了第一个块）
                if i > 0 and chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = (
                        prev_chunk[-cover_content:]
                        if len(prev_chunk) > cover_content
                        else prev_chunk
                    )
                    chunk_part = cover_part + chunk_part

                chunk_text.append(chunk_part)

            # 重置当前块状态
            curr_chunk = ""
            curr_len = 0

        elif curr_len + line_len + 1 <= token_len:  # +1 for newline
            # 当前行可以加入当前块
            if curr_chunk:
                curr_chunk += "\n"
                curr_len += 1
            curr_chunk += line
            curr_len += line_len
        else:
            # 当前行无法加入当前块，开始新块
            if curr_chunk:
                chunk_text.append(curr_chunk)

            # 开始新块，添加覆盖内容
            if chunk_text:
                prev_chunk = chunk_text[-1]
                cover_part = (
                    prev_chunk[-cover_content:]
                    if len(prev_chunk) > cover_content
                    else prev_chunk
                )
                curr_chunk = cover_part + "\n" + line
                curr_len = len(enc.encode(cover_part)) + 1 + line_len
            else:
                curr_chunk = line
                curr_len = line_len

    # 添加最后一个块（如果有内容）
    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text
