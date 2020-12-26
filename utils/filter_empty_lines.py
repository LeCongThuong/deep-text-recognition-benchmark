import re


def read_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().splitlines()
    return content


def write_to_file(corpus, dest_path):
    with open(dest_path, 'w', encoding='utf-8') as f:
        for item in corpus:
            f.write("%s\n" % item)


def filter_emtpy_lines(content, character_vocab):
    out_of_vocab = f'[^{character_vocab}]'
    count = 0
    filtered_content = []
    for line in content:
        print(f'\r{line}', end='')
        filtered_line = re.sub(out_of_vocab, '', line)
        if len(filtered_line) == 0:
            count = count + 1
        else:
            filtered_content.extend(line)

    print("Done")
    print("Num of invalid lines: ", count)
    return filtered_content


def main():
    file_path = '/home/love_you/ocr-gen/vi.txt'
    character_vocab = 'hjbóẺoÝLvÚẼÁÂẩởĨỈtgKứẾmŨÒWsăỷịơIÔỀửãùaXP9ẰẳỉẹỶzầẪâỸỎảệyOựỬẵỘxCỐlỲD6ộỦỒĂƠÌồ1áTFnỆpHẽờếỏẢYẨUắƯẦíÃẤJèýẲ2i4ẬỊÊÓớR7ÙÕàGỨềỳecêSéừqQạòấỮ0ốẫ5õfỗđỡúNũỤợỖỠMằẸôỚặuỌỞụÀEkĐÉBưẮ3ỂễAìỜủỔỢổọwậdZĩẻ8ỄỰểrÈẴÍỪẶẠữỹV '
    dest_path = '/home/love_you/ocr-gen/filtered_vi.txt'
    content = read_from_file(file_path)
    filtered_content = filter_emtpy_lines(content, character_vocab)
    write_to_file(filtered_content, dest_path)


main()
