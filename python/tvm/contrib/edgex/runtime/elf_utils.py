# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""ELF Utilities"""
# pylint: disable-msg=C0103,C0116
import os
import struct
import re
import tempfile
import tvm
from tvm.contrib.edgex.base.edgexlog import EdgexLog as el
from .edgex_runtime import get_max_pm_size, get_iss_start_pc


class ELFHeader:
    """ELF header struct"""

    def __init__(self):
        self.e_type = 1  # ET_REL
        self.e_machine = 0xFC
        self.e_version = 1  # EV_CURRENT
        self.e_entry = 0  # no entry
        self.e_flags = 0
        self.e_ehsize = 52

        self.program_header_num = 0
        self.program_header_size = 0
        self.program_header_offset = 0

        self.section_num = 0
        self.section_size = 40
        self.section_header_offset = 0
        self.string_table_index = 1

    def dump(self):
        return struct.pack(
            "16c2H5I6H",
            b"\x7f",
            b"E",
            b"L",
            b"F",
            b"\x01",  # EI_CLASS=ELFCLASS32
            b"\x01",  # EI_DATA=ELFDATA2LSB, Two's complement, little-endian.
            b"\x01",  # EI_VERSION=EV_CURRENT
            b"\x00",  # EI_OSABI=0
            b"\x00",  # EI_ABIVERSION
            b"\x00",
            b"\x00",
            b"\x00",
            b"\x00",
            b"\x00",
            b"\x00",
            b"\x00",
            self.e_type,
            self.e_machine,
            self.e_version,
            self.e_entry,
            self.program_header_offset,
            self.section_header_offset,
            self.e_flags,
            self.e_ehsize,
            self.program_header_size,
            self.program_header_num,
            self.section_size,
            self.section_num,
            self.string_table_index,
        )

    @staticmethod
    def load(data, offset):
        header = ELFHeader()
        (
            magic16,
            header.e_type,
            header.e_machine,
            header.e_version,
            header.e_entry,
            header.program_header_offset,
            header.section_header_offset,
            header.e_flags,
            header.e_ehsize,
            header.program_header_size,
            header.program_header_num,
            header.section_size,
            header.section_num,
            header.string_table_index,
        ) = struct.unpack("16s2H5I6H", data[offset : offset + 52])
        if magic16 != b"\x7fELF\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00":
            raise ValueError(f"Unsupported elf header: {magic16}")
        return header


class ELFSection:
    "ELF section struct"
    SHT_NULL = 0
    SHT_PROGBITS = 1
    SHT_SYMTAB = 2
    SHT_STRTAB = 3

    SHF_WRITE = 0x1
    SHF_ALLOC = 0x2
    SHF_EXECINSTR = 0x4
    SHF_MASKPROC = 0xF0000000

    def __init__(
        self, name, strtab_name_offset, stype, flags, link, info, addralign, entsize, content
    ):
        self.name = name
        self.strtab_name_offset = strtab_name_offset
        self.stype = stype
        self.flags = flags
        self.addr = 0
        self.offset = 0
        self.size = 0 if content is None else len(content)
        self.link = link
        self.info = info
        self.addralign = addralign
        self.entsize = entsize
        if content and not isinstance(content, bytearray):
            content = bytearray(content)
        self.content = content

    def set_content(self, content):
        self.content = bytearray(content)
        self.size = len(self.content)

    def dump_content(self):
        return b"" if self.content is None else self.content

    def dump_entry(self):
        return struct.pack(
            "10I",
            self.strtab_name_offset,  # sh_name offset,
            self.stype,  # sh_type=SHT_NULL
            self.flags,  # sh_flags
            self.addr,  # sh_addr
            self.offset,  # sh_offset
            self.size,  # sh_size
            self.link,  # sh_link
            self.info,  # sh_info
            self.addralign,  # sh_addralign
            self.entsize,  # sh_entsize
        )

    @staticmethod
    def load(data, offset, strtab_offset):
        (
            strtab_name_offset,
            stype,
            flags,
            addr,
            data_offset,
            size,
            link,
            info,
            addralign,
            entsize,
        ) = struct.unpack("10I", data[offset : offset + 40])
        k = strtab_offset + strtab_name_offset
        while data[k] != 0 and k < len(data):
            k += 1
        name = data[strtab_offset + strtab_name_offset : k].decode("ascii")
        content = data[data_offset : data_offset + size]
        section = ELFSection(
            name, strtab_name_offset, stype, flags, link, info, addralign, entsize, content
        )
        section.addr = addr
        section.offset = data_offset
        section.size = size
        return section


class ELFSymbol:
    """ELF symbol struct"""

    STB_GLOBAL = 1

    STT_NOTYPE = 0

    def __init__(self, name, strtab_name_offset, value, size, bind, stype, other, section_idx):
        self.name = name
        self.strtab_name_offset = strtab_name_offset
        self.value = value
        self.size = size
        self.bind = bind
        self.stype = stype
        self.other = other
        self.section_idx = section_idx

    def dump(self):
        return struct.pack(
            "3IBBH",
            self.strtab_name_offset,  # st_name offset,
            self.value,  # st_value
            self.size,  # st_size
            (self.bind << 4) + (self.stype & 0x0F),  # st_info
            self.other,  # st_other
            self.section_idx,  # st_shndx
        )

    @staticmethod
    def load(data, offset, strtab_offset):
        strtab_name_offset, value, size, info, other, section_idx = struct.unpack(
            "3IBBH", data[offset : offset + 16]
        )
        bind = info >> 4
        stype = info & 0x0F
        k = strtab_offset + strtab_name_offset
        while data[k] != 0 and k < len(data):
            k += 1
        name = data[strtab_offset + strtab_name_offset : k].decode("ascii")
        symbol = ELFSymbol(name, strtab_name_offset, value, size, bind, stype, other, section_idx)
        return symbol


class ELFObject:
    """ELF object struct"""

    def __init__(self, data=None, offset=0):
        if data:
            self.parse_elf(data, offset)
            return
        self.header = ELFHeader()
        self.sections = []
        self.symbols = []
        null_section = ELFSection(
            "",
            0,
            stype=ELFSection.SHT_NULL,
            flags=0,
            link=0,
            info=0,
            addralign=0,
            entsize=0,
            content=None,
        )
        self.sections.append(null_section)
        strtab_section = ELFSection(
            ".strtab",
            1,
            stype=ELFSection.SHT_STRTAB,
            flags=0,
            link=0,
            info=0,
            addralign=1,
            entsize=0,
            content=None,
        )
        self.strings = [".strtab"]
        self.strings_offset = 2 + len(".strtab")
        self.sections.append(strtab_section)
        self.header.section_num = 2

    @staticmethod
    def load(data, offset):
        return ELFObject(data, offset)

    def parse_elf(self, data, offset):
        self.header = ELFHeader.load(data, offset)
        offset += self.header.e_ehsize
        if self.header.string_table_index <= 0:
            raise ValueError(f"Do not support string table index: {self.header.string_table_index}")
        self.sections = []
        strtab_offset = (
            self.header.section_header_offset
            + self.header.section_size * self.header.string_table_index
        )
        (strtab_offset,) = struct.unpack("I", data[strtab_offset + 16 : strtab_offset + 20])
        section_entry_offset = self.header.section_header_offset
        symtab_index = -1
        for i in range(self.header.section_num):
            section = ELFSection.load(data, section_entry_offset, strtab_offset)
            self.sections.append(section)
            section_entry_offset += self.header.section_size
            if section.name == ".symtab":
                symtab_index = i

        symtab = self.sections[symtab_index]
        self.symbols = []
        for i in range(symtab.size // 16):
            symbol = ELFSymbol.load(data, symtab.offset + i * 16, strtab_offset)
            self.symbols.append(symbol)

    def add_section(self, name, stype, flags, link, info, addralign, entsize, content):
        section = ELFSection(
            name, self.strings_offset, stype, flags, link, info, addralign, entsize, content
        )
        self.strings.append(name)
        self.sections.append(section)
        self.header.section_num += 1
        self.strings_offset += len(name) + 1

    def get_section(self, name):
        for idx, section in enumerate(self.sections):
            if section.name == name:
                return idx, section
        return None, None

    def add_symbol(self, name, value, size, bind, stype, other, section_idx):
        symbol = ELFSymbol(name, self.strings_offset, value, size, bind, stype, other, section_idx)
        self.symbols.append(symbol)
        self.strings.append(name)
        self.strings_offset += len(name) + 1

    def finish(self):
        if len(self.symbols) > 0:
            content = b""
            for sym in self.symbols:
                content += sym.dump()
            self.add_section(
                ".symtab",
                stype=ELFSection.SHT_SYMTAB,
                flags=0,
                link=1,
                info=0,
                addralign=4,
                entsize=16,
                content=content,
            )

        section_info_offset = 52
        for sec in self.sections[2:]:
            if sec.addralign > 0:
                section_info_offset = (
                    (section_info_offset + sec.addralign - 1) // sec.addralign
                ) * sec.addralign
                sec.offset = section_info_offset
                section_info_offset += sec.size

        # process strtab
        strtab_section = self.sections[1]
        strtab_section.offset = section_info_offset
        strtab_section.set_content(
            b"\x00" + b"\x00".join([bytes(_.encode("ascii")) for _ in self.strings]) + b"\x00"
        )
        self.header.section_header_offset = section_info_offset + strtab_section.size

    def dump(self):
        result = self.header.dump()
        result += self.sections[0].dump_content()
        for sec in self.sections[2:]:
            result += sec.dump_content()
        result += self.sections[1].dump_content()
        for sec in self.sections:
            result += sec.dump_entry()
        return result


class LstFileData:
    """Record full information of lst file"""

    label_tab_start0 = "// Branch Labels"
    label_tab_start1 = "//=============="
    label_pattern = re.compile(
        r"// ([^ ]+)[ ]+([^ ]+)[ ]+(\d+)[ ]+([0-9a-fA-F]+)[ ]+([0-9a-fA-F]+).*"
    )
    label_pattern = re.compile(
        r"// ([^ ]+)[ ]+([^ ]+)[ ]+(\d+)[ ]+([0-9a-fA-F]+)[ ]+([0-9a-fA-F]+)"
    )
    line_pattern = re.compile(
        r"(([ ]{64})|([0-9a-fA-F]{64}))//(\d+)[ ]+:(([0-9a-fA-F]{8}):[ ]+(\d+))?[ ]+(.*)"
    )

    def __init__(self, lst_file):
        line_cnt = 0
        self.labels = []
        self.lines = []
        self.valid_lines = []
        parse_labels = False
        for line in open(lst_file):
            line_cnt += 1
            if line.strip() == "":
                continue

            if line.startswith(LstFileData.label_tab_start0) or (
                parse_labels and line.startswith(LstFileData.label_tab_start1)
            ):
                parse_labels = True
                continue

            if parse_labels:
                m = LstFileData.label_pattern.match(line)
                if m is None:
                    raise ValueError(f"Corrupted lst file {lst_file} at line {line_cnt}: {line}")
                name = m.group(1)
                filename = m.group(2)
                lineno = int(m.group(3))
                pc = int(m.group(4), 16)
                ext_offset = m.group(5)
                self.labels.append((name, filename, lineno, pc, ext_offset))
                continue

            m = LstFileData.line_pattern.match(line)
            if m is None:
                raise ValueError(f"Corrupted lst file {lst_file} at line {line_cnt}: {line}")
            inst = m.group(3)
            is_valid = inst is not None
            lineno = int(m.group(4))
            pc = int(m.group(6), 16) if is_valid else None
            valid_lineno = int(m.group(7)) if is_valid else None
            text = m.group(8)
            if is_valid:
                self.valid_lines.append(len(self.lines))
            self.lines.append((inst, lineno, pc, valid_lineno, text))

        self.inst_cnt = []
        prev_addr = None
        for idx in self.valid_lines:
            line = self.lines[idx]
            if prev_addr is not None:
                self.inst_cnt.append(line[2] - prev_addr)
            assert line[2] is not None
            prev_addr = line[2]
        last_inst = self.lines[self.valid_lines[-1]][0]
        self.inst_cnt.append(
            sum([1 if last_inst[k * 8 : k * 8 + 8] != "00000000" else 0 for k in range(8)])
        )


def create_relocatable_object(bin_file, lst_file, output_path):
    """Create relocatable object from bin and lst.

    Parameters:
    -----------
    bin_file : str
        bin file path

    lst_file : str
        lst file path

    output_path : str
        output obj file path
    """
    lst_data = LstFileData(lst_file)
    start_pc = lst_data.lines[lst_data.valid_lines[0]][2]

    with open(bin_file, "rb") as input_file:
        bin_data = input_file.read()
    if len(bin_data) % 4 != 0:
        raise ValueError(f"Corrupted bin file: {bin_file}")

    bin_offset = 0
    for i, line_idx in enumerate(lst_data.valid_lines):
        inst, lineno, _, _, text = lst_data.lines[line_idx]
        inst_cnt = lst_data.inst_cnt[i]
        for k in range(inst_cnt):
            (bin_inst,) = struct.unpack("I", bin_data[bin_offset : bin_offset + 4])
            bin_inst = hex(bin_inst)[2:].rjust(8, "0")
            bin_offset += 4
            hex_inst = inst[64 - k * 8 - 8 : 64 - k * 8]
            if hex_inst != bin_inst:
                raise ValueError(
                    f"Inconsistent instruction encoding at line {lineno}'s {k}th inst\n"
                    + f"  bin: ${bin_file} {bin_inst}\n  lst: {lst_file} {hex_inst}: {text}"
                )

    with open(output_path, "wb") as out_file:
        obj = ELFObject()
        obj.add_section(
            ".text",
            stype=ELFSection.SHT_PROGBITS,
            flags=ELFSection.SHF_EXECINSTR | ELFSection.SHF_ALLOC,
            link=0,
            info=0,
            addralign=4,
            entsize=0,
            content=bin_data,
        )
        for name, _, lineno, pc, _ in lst_data.labels:
            obj.add_symbol(
                name,
                value=4 * (pc - start_pc),
                size=0,
                bind=ELFSymbol.STB_GLOBAL,
                stype=ELFSymbol.STT_NOTYPE,
                other=0,
                section_idx=2,
            )
        obj.finish()
        out_file.write(obj.dump())
    with open(output_path, "rb") as in_file:
        obj = ELFObject.load(in_file.read(), 0)
    with open(output_path, "wb") as out_file:
        out_file.write(obj.dump())


def merge_relocatable_object(main_objects, lib_objects, output_path):
    """Merge relocatable objects and resolve b/bl relocations.
    The result .text section layout would be
    main_text1
    main_text2
    ...
    lib_text1
    lib_text2
    ...

    Parameters:
    -----------
    main_objects : list of str or bytes
        objects which need relocation

    lib_objects : list of str or bytes
        objects served as a library obj, do not do relocation

    output_path : str
        output obj file path
    """
    result_object = ELFObject()
    merged_bin = bytearray()
    text_offset = 0
    symbol_offsets = {}  # symbol_name -> offset in bytes
    symbol_sizes = {}  # symbol_name -> size in bytes
    unresolved = {}  # symbol_name -> [reloc positions]

    def do_reloc(bin_data, symbol_name, reloc_pos, symbol_offset):
        if symbol_offset % 4 != 0:
            raise ValueError(f"Illegal symbol offset: {symbol_offset}, not aligned with 4")
        symbol_addr_line = symbol_offset // 4
        if len(bin_data) < reloc_pos + 4:
            raise ValueError(
                f"Illegal reloc position: {reloc_pos} for {symbol_name}, exceed bin size"
            )
        (inst,) = struct.unpack("I", bin_data[reloc_pos : reloc_pos + 4])
        if not inst & 0x3F in [0x22, 0x20]:
            raise ValueError(f"Illegal reloc position {reloc_pos}: not b or bl inst")
        new_inst = (inst & 0x1FFF) | ((symbol_addr_line - reloc_pos // 4) << 13)
        bin_data[reloc_pos : reloc_pos + 4] = struct.pack("I", new_inst)

    def process_obj(obj):
        nonlocal merged_bin, text_offset
        if isinstance(obj, str) and os.path.isfile(obj):
            with open(obj, "rb") as in_file:
                obj = in_file.read()
        assert isinstance(obj, (bytes, bytearray))
        elf_object = ELFObject.load(obj, 0)

        _, text_section = elf_object.get_section(".text")
        if not text_section:
            return
        bin_data = text_section.content

        # lookup reloc information
        _, reloc_section = elf_object.get_section(".rel.text")
        if reloc_section:
            reloc_entry_cnt = reloc_section.size // 8
            for i in range(reloc_entry_cnt):
                reloc_position, r_info = struct.unpack(
                    "II", reloc_section.content[i * 8 : i * 8 + 8]
                )
                symbol_index = r_info >> 8
                reloc_type = r_info & 0xFF
                if reloc_type != 0x2D:
                    continue
                symbol_name = elf_object.symbols[symbol_index].name
                if symbol_name in symbol_offsets:
                    offset = symbol_offsets[symbol_name]
                    do_reloc(merged_bin, symbol_name, reloc_position, offset)
                else:
                    if symbol_name not in unresolved:
                        unresolved[symbol_name] = []
                    unresolved[symbol_name].append(text_offset + reloc_position)

        # merge bin
        merged_bin += bin_data

        # lookup symbol defs
        for symbol in elf_object.symbols:
            # skip non-global symbol
            if symbol.bind != ELFSymbol.STB_GLOBAL or symbol.section_idx <= 0:
                continue
            if symbol.name in symbol_offsets:
                raise ValueError(f"Symbol redefinition: {symbol.name}")
            offset = text_offset + symbol.value
            symbol_offsets[symbol.name] = offset
            symbol_sizes[symbol.name] = symbol.size

            if symbol.name in unresolved:
                for reloc_position in unresolved[symbol.name]:
                    do_reloc(merged_bin, symbol.name, reloc_position, offset)
                unresolved.pop(symbol.name)
        text_offset += len(bin_data)

    for obj in main_objects:
        process_obj(obj)
    if len(unresolved) > 0:
        for obj in lib_objects:
            process_obj(obj)

    if len(unresolved) > 0:
        raise ValueError(f"Unresolved symbols: {', '.join(unresolved.keys())}")

    result_object.add_section(
        ".text",
        stype=ELFSection.SHT_PROGBITS,
        flags=ELFSection.SHF_EXECINSTR | ELFSection.SHF_ALLOC,
        link=0,
        info=0,
        addralign=4,
        entsize=0,
        content=merged_bin,
    )
    for symbol_name in symbol_offsets:
        result_object.add_symbol(
            symbol_name,
            symbol_offsets[symbol_name],
            symbol_sizes[symbol_name],
            ELFSymbol.STB_GLOBAL,
            ELFSymbol.STT_NOTYPE,
            0,
            2,
        )
    result_object.finish()
    result_data = result_object.dump()
    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(output_path, "wb") as out_file:
            out_file.write(result_data)
    return result_data


def wrap_obj_as_single_kernel(obj, kernel_name, output_path):
    """Wrap a relocatable object as an object with single kernel function start from 0x0
    Parameters:
    -----------
    obj : str or bytes
        object file path or object data bytes

    kernel_name : str
        kernel function name

    output_path : str
        output obj file path
    """
    if isinstance(obj, str) and os.path.isfile(obj):
        with open(obj, "rb") as in_file:
            obj = in_file.read()
    assert isinstance(obj, bytes)
    input_object = ELFObject.load(obj, 0)
    output_object = ELFObject()

    section_idx, text_section = input_object.get_section(".text")
    assert text_section
    if text_section:
        output_object.add_section(
            ".text",
            text_section.stype,
            text_section.flags,
            text_section.link,
            text_section.info,
            text_section.addralign,
            text_section.entsize,
            text_section.content,
        )
        output_object.add_symbol(
            kernel_name,
            0,
            len(text_section.content),
            ELFSymbol.STB_GLOBAL,
            ELFSymbol.STT_NOTYPE,
            0,
            section_idx,
        )
    output_object.finish()
    result_data = output_object.dump()
    if output_path:
        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(output_path, "wb") as out_file:
            out_file.write(result_data)
    return result_data


@tvm._ffi.register_func("tvm.edgex.get_linked_obj")
def get_linked_obj(obj, kernel_name, output_dir, as_single_kernel=False):
    """Helper to get linked object from llvm-gen relocatable object

    Parameters:
    -----------
    obj : str or bytes
        object file path or object data bytes

    kernel_name : str
        kernel function name

    output_dir : str
        directory to put intermedia files

    as_single_kernel : bool
        whether wrap result as a single kernel
    """
    if isinstance(obj, str) and os.path.isfile(obj):
        with open(obj, "rb") as in_file:
            obj = in_file.read()
    assert isinstance(obj, (bytes, bytearray))

    tmp_file_dir = None
    if output_dir is None or output_dir.strip() == "":
        tmp_file_dir = tempfile.mkdtemp(prefix="/tmp/edgex_linker_workspace_")
        output_dir = tmp_file_dir
    else:
        output_dir = os.path.join(output_dir, kernel_name, "linker")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    vu_resource_libs = []
    dcl_root_dir = os.environ.get("EDGEX_ROOT_DIR", "./")
    invoke_ass = tvm.get_global_func("tvm.edgex.invoke_assembler")
    for lib_name in ["get_mbx_lock", "cfg_vu_desp"]:
        ass_dir = invoke_ass(
            lib_name, os.path.join(dcl_root_dir, "ass", f"{lib_name}.asm"), 0, output_dir, True
        )
        objpath = os.path.join(ass_dir, f"{lib_name}.obj")
        create_relocatable_object(
            os.path.join(ass_dir, f"{lib_name}.bin"),
            os.path.join(ass_dir, f"{lib_name}_cpp.lst"),
            objpath,
        )
        vu_resource_libs.append(objpath)

    bin_path = os.path.join(output_dir, f"{kernel_name}.obj")
    res = merge_relocatable_object([obj], vu_resource_libs, bin_path)
    if as_single_kernel:
        bin_path = os.path.join(output_dir, f"{kernel_name}.single.obj")
        res = wrap_obj_as_single_kernel(res, kernel_name, bin_path)
    return res


@tvm._ffi.register_func("tvm.edgex.extract_bin_data")
def extract_bin_data(obj, kernel_name):
    """Helper to get bin data from relocatable object by kernel function name

    Parameters:
    -----------
    obj : str or bytes
        object file path or object data bytes

    kernel_name : str
        kernel function name
    """
    if isinstance(obj, str) and os.path.isfile(obj):
        with open(obj, "rb") as in_file:
            obj = in_file.read()
    assert isinstance(obj, (bytes, bytearray))
    elf_obj = ELFObject(obj)
    text_section_idx, text = elf_obj.get_section(".text")
    if text is None:
        raise ValueError(f"No .text section for {kernel_name}")
    for symbol in elf_obj.symbols:
        if symbol.name != kernel_name:
            continue
        if symbol.size <= 0:
            raise ValueError(f"Illegal size {symbol.size} for {kernel_name}")
        if symbol.bind != ELFSymbol.STB_GLOBAL:
            raise ValueError(f"{kernel_name} is not a global symbol")
        if symbol.section_idx != text_section_idx:
            raise ValueError(f"{kernel_name} do not refer to .text section")
        if symbol.value < 0:
            raise ValueError(f"Illegal start offset of {kernel_name}: {symbol.value}")
        if symbol.value + symbol.size > len(text.content):
            raise ValueError(f"Offset out of bound of {kernel_name}: {symbol.value + symbol.size}")
        return text.content[symbol.value : symbol.value + symbol.size]

    raise ValueError(f"Can not find kernel function {kernel_name}")


@tvm._ffi.register_func("tvm.edgex.bin2lst")
def bin2lst(bin_data, start_pc):
    """Helper to create a dummy lst file from bin
    Parameters:
    -----------
    bin_data : str or bytes
        bin file path or bin data

    start_pc : int
        expected start pc counter for bin

    Returns:
    --------
        lst string
    """
    if isinstance(bin_data, str) and os.path.isfile(bin_data):
        with open(bin_data, "rb") as in_file:
            bin_data = in_file.read()
    lst_str = ""
    bin_offset = 0
    pc = start_pc
    if len(bin_data) > get_max_pm_size():
        pc = get_iss_start_pc("icache")
        el.i("icache mode enabled")
    lines = []
    while bin_offset < len(bin_data):
        inst_list = []
        do_fetch = True
        line_pc = pc
        while do_fetch:
            do_fetch = False
            if bin_offset + 4 > len(bin_data):
                raise ValueError(f"Bin file corrupt at {hex(bin_offset)}")
            (inst_part1,) = struct.unpack("I", bin_data[bin_offset : bin_offset + 4])
            is_cu = inst_part1 & 0x01 == 0
            multi_issue = inst_part1 & 0x100 > 0
            if is_cu:
                bin_offset += 4
                pc = pc + 1
                inst = hex(inst_part1)[2:].rjust(8, "0")
            else:
                if bin_offset + 8 > len(bin_data):
                    raise ValueError(f"Bin file corrupt at {hex(bin_offset)}")
                (inst_part2,) = struct.unpack("I", bin_data[bin_offset + 4 : bin_offset + 8])
                inst = hex(inst_part2)[2:].rjust(8, "0") + hex(inst_part1)[2:].rjust(8, "0")
                bin_offset += 8
                pc = pc + 2
            inst_list.append(inst)
            if multi_issue:
                do_fetch = True
                if len(inst_list) >= 4:
                    raise ValueError(
                        f"Bin file corrupt at {hex(bin_offset)}, "
                        + ">4 parallel issue should be not possible"
                    )
        lines.append((inst_list, line_pc))

    for i, (inst_list, pc) in enumerate(lines):
        lineno = i + 1
        lineno_enc = str(lineno).ljust(9, " ")
        inst_enc = "".join(reversed(inst_list)).rjust(64, "0")
        pc_enc = hex(pc)[2:].rjust(8, "0")
        line = f"{inst_enc}//{lineno_enc}:{pc_enc}:  {lineno}          <Unknown>\n"
        lst_str += line

    lst_str += "\n"
    lst_str += "// Branch Labels                                                              Src File                              Line No                     pc           ext_offt\n"  # pylint:disable=C0301
    lst_str += "//===================================================================================================================================================================\n"  # pylint:disable=C0301
    return lst_str


@tvm._ffi.register_func("tvm.edgex.create_full_kernels_obj")
def merge_relocatable_object_wrapper(output_dir, *objs):
    """Create an elf object contains all function kernels"""
    output_path = None
    if output_dir and output_dir.strip() != "":
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "full_kernels.obj")
    return merge_relocatable_object(objs, [], output_path)
