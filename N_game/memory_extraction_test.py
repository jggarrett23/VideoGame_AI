import ctypes
import os
import win32gui
import win32con
import win32api
import win32process
import time
import pymem
import pymem.process

# Constants from Windows API
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_VM_OPERATION = 0x0008
PROCESS_ALL_ACCESS = 0x1F0FFF


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('BaseAddress', ctypes.c_void_p),
        ('AllocationBase', ctypes.c_void_p),
        ('AllocationProtect', ctypes.c_uint32),
        ('RegionSize', ctypes.c_size_t),
        ('State', ctypes.c_uint32),
        ('Protect', ctypes.c_uint32),
        ('Type', ctypes.c_uint32),
    ]


def open_process(pid):
    return ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False,
                                              pid)


def close_handle(handle):
    ctypes.windll.kernel32.CloseHandle(handle)


def read_memory_from_process(pid, address, size):
    process_handle = open_process(pid)

    if process_handle:
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t()

        if ctypes.windll.kernel32.ReadProcessMemory(process_handle, address, buffer, size, ctypes.byref(bytes_read)):
            print(f"Read {bytes_read.value} bytes from memory:", buffer.raw)
        else:
            print("Failed to read memory")
        close_handle(process_handle)
    else:
        print("Failed to open process")


# Define function to enumerate memory regions
def enumerate_memory_regions(pid):
    process_handle = open_process(pid)
    if process_handle:
        address = 0
        while True:
            mbi = MEMORY_BASIC_INFORMATION()
            result = ctypes.windll.kernel32.VirtualQueryEx(process_handle, address, ctypes.byref(mbi),
                                                           ctypes.sizeof(mbi))
            print(f"VirtualQueryEx result: {result}")
            if result == 0:
                error_code = ctypes.windll.kernel32.GetLastError()
                if error_code != 0x18:  # Error_NO_MORE_FILES
                    print(f"VirtualQueryEx failed with error code: {error_code}")
                break
            if mbi.BaseAddress:
                print(f"BaseAddress: {mbi.BaseAddress}")
                address = mbi.BaseAddress.value + mbi.RegionSize
            else:
                print("BaseAddress is None")
                break
        close_handle(process_handle)
    else:
        print("Failed to open process")


if __name__ == '__main__':

    os.startfile('D:\\VideoGame_AI\\N_game\\Nv2-PC.exe')
    time.sleep(5)

    hwnd = win32gui.FindWindow(None, 'Adobe Flash Player 11')

    if hwnd:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        if pid:
            pm = pymem.Pymem(pid)

            modules = pm.list_modules()

            for module in modules:
                if module.name == 'Nv2-PC.exe':
                    base_address = module.lpBaseOfDll
                    size = module.SizeOfImage
                    break

            buffer_size = 1024
            time_left_offset = 0x0084D188
            time_left_offset_chain = [0x484, 0xB8, 0x0, 0x53C, 0x4, 0x2C, 0x38]

            # intial pointer address
            time_left_address = base_address + time_left_offset

            # traverse pointer chain
            for offset in time_left_offset_chain:
                time_left_address = pm.read_int(time_left_address) + offset

            while True:
                time_left = pm.read_int(time_left_address) / 60
                print(time_left/60)

        else:
            print("Failed to get process ID")
    else:
        print('Failed to find window')

