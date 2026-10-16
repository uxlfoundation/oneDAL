/*******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// Moves the DLL exports of Windows objects from the objects into a module-definition file.
//
// The objects are compiled with `__declspec(dllexport)`, which makes the compiler emit every
// member of an exported class and add an `/EXPORT:` linker directive for each exported symbol.
// The static libraries archive the same objects, and the linker applies those directives to any
// EXE or DLL that pulls an object in. So right after compiling, the directives are moved out:
//
//   coff_exports move <object> <list>    writes the object's exports to <list>, one per line
//                                        ("name" or "name DATA"), and overwrites each `/EXPORT:`
//                                        directive in the object with spaces
//   coff_exports def <link list> <def>   writes the EXPORTS of the DLL linked from <link list>
//                                        (the objects' paths, one per line) from their lists
//
// Only the bytes of the directives change: section sizes, symbols and the other directives
// (`/FAILIFMISMATCH`, `/DEFAULTLIB`, ...) stay as the compiler wrote them.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t read_u16(const std::vector<char>& buf, std::size_t offset) {
    if (offset + 2 > buf.size()) {
        throw std::runtime_error("truncated COFF header");
    }
    const auto* p = reinterpret_cast<const unsigned char*>(buf.data() + offset);
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8;
}

std::uint32_t read_u32(const std::vector<char>& buf, std::size_t offset) {
    return read_u16(buf, offset) | read_u16(buf, offset + 2) << 16;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/// The file offset and size of every `.drectve` section of a COFF object.
///
/// @param[in] obj The object file contents, regular or `/bigobj` COFF
///
/// @return Pairs of (offset, size) of the raw section data
std::vector<std::pair<std::size_t, std::size_t>> directive_sections(const std::vector<char>& obj) {
    // The /bigobj header starts with IMAGE_FILE_MACHINE_UNKNOWN and 0xFFFF, followed by its
    // class GUID; an ordinary header starts with the machine type
    static const unsigned char bigobj_guid[16] = { 0xC7, 0xA1, 0xBA, 0xD1, 0xEE, 0xBA, 0xa9, 0x4b,
                                                   0xAF, 0x20, 0xFA, 0xF6, 0x6A, 0xA4, 0xDC, 0xB8 };
    std::size_t section_count = 0;
    std::size_t headers = 0;
    if (read_u16(obj, 0) == 0 && read_u16(obj, 2) == 0xFFFF) {
        if (obj.size() < 56 || std::memcmp(obj.data() + 12, bigobj_guid, 16) != 0) {
            throw std::runtime_error("not a COFF object (an LTCG or import object?)");
        }
        section_count = read_u32(obj, 44);
        headers = 56;
    }
    else {
        const std::uint32_t machine = read_u16(obj, 0);
        if (machine != 0x8664 && machine != 0xAA64 && machine != 0xA641 && machine != 0x14C) {
            throw std::runtime_error("not a COFF object for x86 or ARM64");
        }
        section_count = read_u16(obj, 2);
        headers = 20 + read_u16(obj, 16);
    }

    std::vector<std::pair<std::size_t, std::size_t>> sections;
    for (std::size_t i = 0; i < section_count; ++i) {
        const std::size_t header = headers + 40 * i;
        if (header + 40 > obj.size()) {
            throw std::runtime_error("truncated section table");
        }
        if (std::memcmp(obj.data() + header, ".drectve", 8) != 0) {
            continue;
        }
        const std::size_t size = read_u32(obj, header + 16);
        const std::size_t offset = read_u32(obj, header + 20);
        if (offset + size > obj.size()) {
            throw std::runtime_error("truncated .drectve section");
        }
        sections.emplace_back(offset, size);
    }
    return sections;
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\0';
}

bool starts_with_nocase(const char* s, std::size_t n, const char* prefix) {
    const std::size_t len = std::strlen(prefix);
    if (n < len) {
        return false;
    }
    for (std::size_t i = 0; i < len; ++i) {
        const char c = s[i] >= 'A' && s[i] <= 'Z' ? char(s[i] - 'A' + 'a') : s[i];
        if (c != prefix[i]) {
            return false;
        }
    }
    return true;
}

/// Turns the argument of an `/EXPORT:` directive into a module-definition line.
///
/// @param[in] arg The text after `/EXPORT:`, a symbol name in optional quotes and an optional
///                `,DATA`
///
/// @return "name" or "name DATA"
std::string export_line(const std::string& arg) {
    std::string name = arg;
    std::string option;
    std::size_t end = 0;
    if (!arg.empty() && arg[0] == '"') {
        end = arg.find('"', 1);
        if (end == std::string::npos) {
            throw std::runtime_error("unterminated quote in /EXPORT:" + arg);
        }
        name = arg.substr(1, end - 1);
        ++end;
    }
    else {
        end = arg.find(',');
        name = arg.substr(0, end);
    }
    if (end != std::string::npos && end < arg.size()) {
        if (arg[end] != ',') {
            throw std::runtime_error("unexpected text in /EXPORT:" + arg);
        }
        option = arg.substr(end + 1);
    }
    // Aliases, ordinals and PRIVATE never come from dllexport; refuse rather than mistranslate
    if (name.empty() || name.find_first_of(" \t=,;\"") != std::string::npos ||
        !(option.empty() || option == "DATA" || option == "data")) {
        throw std::runtime_error("unsupported /EXPORT:" + arg);
    }
    return option.empty() ? name : name + " DATA";
}

void write_file(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << text;
        if (!out.flush()) {
            throw std::runtime_error("cannot write " + tmp);
        }
    }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("cannot rename " + tmp + " to " + path);
    }
}

void move_exports(const std::string& obj_path, const std::string& list_path) {
    // A list left over from an earlier compile must not survive a failure below: without it the
    // DLL link fails loudly instead of exporting the old object's symbols
    std::remove(list_path.c_str());

    std::vector<char> obj = read_file(obj_path);
    std::string list;
    bool changed = false;
    for (const auto& section : directive_sections(obj)) {
        char* text = obj.data() + section.first;
        std::size_t pos = 0;
        const std::size_t size = section.second;
        // A UTF-8 byte order mark may precede the directives
        if (size >= 3 && std::memcmp(text, "\xEF\xBB\xBF", 3) == 0) {
            pos = 3;
        }
        while (pos < size) {
            while (pos < size && is_space(text[pos])) {
                ++pos;
            }
            const std::size_t begin = pos;
            bool quoted = false;
            while (pos < size && (quoted || !is_space(text[pos]))) {
                quoted ^= text[pos] == '"';
                ++pos;
            }
            const char* token = text + begin;
            const std::size_t length = pos - begin;
            if (starts_with_nocase(token, length, "/export:") ||
                starts_with_nocase(token, length, "-export:")) {
                list += export_line(std::string(token + 8, length - 8)) + "\n";
                std::memset(text + begin, ' ', length);
                changed = true;
            }
        }
    }

    if (changed) {
        std::fstream out(obj_path, std::ios::binary | std::ios::in | std::ios::out);
        for (const auto& section : directive_sections(obj)) {
            out.seekp(std::streamoff(section.first));
            out.write(obj.data() + section.first, std::streamsize(section.second));
        }
        if (!out.flush()) {
            throw std::runtime_error("cannot write " + obj_path);
        }
    }
    write_file(list_path, list);
}

void write_def(const std::string& link_list_path, const std::string& def_path) {
    std::ifstream link_list(link_list_path);
    if (!link_list) {
        throw std::runtime_error("cannot open " + link_list_path);
    }
    // std::map keeps the output sorted, so the file depends only on the set of exports
    std::map<std::string, bool> exports;
    std::string path;
    while (std::getline(link_list, path)) {
        while (!path.empty() && is_space(path.back())) {
            path.pop_back();
        }
        const std::size_t dot = path.rfind('.');
        if (dot == std::string::npos || path.substr(dot) != ".obj") {
            continue; // resources and libraries carry no exports
        }
        const std::string list_path = path.substr(0, dot) + ".exports";
        std::ifstream list(list_path);
        if (!list) {
            throw std::runtime_error("missing " + list_path + "; delete " + path + " to rebuild it");
        }
        std::string line;
        while (std::getline(list, line)) {
            const bool data = line.size() > 5 && line.compare(line.size() - 5, 5, " DATA") == 0;
            const std::string name = data ? line.substr(0, line.size() - 5) : line;
            const auto it = exports.emplace(name, data).first;
            if (it->second != data) {
                throw std::runtime_error(name + " is exported both as code and as data");
            }
        }
    }

    std::string def = "EXPORTS\n";
    for (const auto& e : exports) {
        def += "    " + e.first + (e.second ? " DATA\n" : "\n");
    }
    write_file(def_path, def);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc == 4 ? argv[1] : "";
        if (mode == "move") {
            move_exports(argv[2], argv[3]);
        }
        else if (mode == "def") {
            write_def(argv[2], argv[3]);
        }
        else {
            std::cerr << "usage: coff_exports move <object> <list> | def <link list> <def>\n";
            return 2;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "coff_exports: " << (argc > 2 ? argv[2] : "") << ": " << e.what() << "\n";
        return 1;
    }
    return 0;
}
