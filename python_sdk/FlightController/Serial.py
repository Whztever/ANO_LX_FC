from sys import byteorder as sysByteorder

import serial


class FC_Serial:
    def __init__(self, port, baudrate, timeout=0.5, byteOrder=sysByteorder):
        """
        初始化方法，用于创建 FC_Serial 对象并打开串口连接。

        Args:
            port (str): 串口名称。
            baudrate (int): 波特率。
            timeout (float, optional): 超时时间。默认为 0.5 秒。
            byteOrder (str, optional): 字节序。默认为系统字节序。

        Returns:
            None
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.data = bytes()
        self._read_buffer = bytes()
        self._in_waiting_buffer = bytes()
        self._reading_flag = False
        self._pack_count = 0
        self._pack_length = 0
        self._byte_order = byteOrder
        self.send_config()
        self.read_config()

    def send_config(self, startBit=[], optionBit=[]):
        """
        设置发送配置。

        Args:
            startBit (list, optional): 发送起始位。默认为空列表。
            optionBit (list, optional): 发送选项位。默认为空列表。

        Returns:
            None
        """
        self._send_start_bit = startBit
        self._send_option_bit = optionBit
        self._send_head = bytes(self._send_start_bit) + bytes(self._send_option_bit)

    def read_config(self, startBit=[]):
        """
        设置读取配置。

        Args:
            startBit (list, optional): 读取起始位。默认为空列表。

        Returns:
            None
        """
        self._read_start_bit = startBit

    def check_rx_data_checksum(self):
        """
        检查接收数据的校验和。

        Returns:
            int: 校验和是否正确。
        """
        length = len(self._read_buffer)
        checksum = 0
        for i in self._read_start_bit:
            checksum += i
            checksum &= 0xFF
        checksum += self._pack_length_bit
        checksum &= 0xFF
        for i in range(0, length):
            checksum += int.from_bytes(
                self._read_buffer[i : i + 1],
                byteorder=self._byte_order,
                signed=False,
            )
            checksum &= 0xFF
        received_checksum = int.from_bytes(self.ser.read(1), byteorder=self._byte_order, signed=False)
        if received_checksum == checksum:
            return 1
        return 0

    def read(self) -> bool:
        """
        读取串口数据。

        Returns:
            bool: 是否读取成功。True 表示读取成功，False 表示读取失败。
        """
        _len = len(self._read_start_bit)
        while self.ser.in_waiting > 0:
            read_byte = self.ser.read(1)
            if not self._reading_flag:
                self._in_waiting_buffer += read_byte
                if len(self._in_waiting_buffer) >= _len and self._in_waiting_buffer[-_len:] == self._read_start_bit:
                    self._reading_flag = True
                    self._read_buffer = bytes()
                    self._pack_count = 0
                    self._pack_length = -1
                    self._in_waiting_buffer = bytes()
            elif self._pack_length == -1:
                self._pack_length_bit = int.from_bytes(read_byte, self._byte_order, signed=False)
                self._pack_length = self._pack_length_bit & 0xFF
            else:
                self._pack_count += 1
                self._read_buffer += read_byte
                if self._pack_count >= self._pack_length:
                    self._reading_flag = False
                    checksum = sum(self._read_buffer[:-1]) & 0xFF
                    checksum += self._pack_length_bit & 0xFF
                    if checksum & 0xFF == self._read_buffer[-1]:
                        self.data = self._read_buffer
                        self._read_buffer = bytes()
                        return True
                    else:
                        self._read_buffer = bytes()
                        return False
        return False

    @property
    def rx_data(self) -> bytes:
        """
        读取接收数据。

        Returns:
            bytes: 接收到的数据。
        """
        return self.data

    def close(self):
        """
        关闭串口连接。

        Returns:
            None
        """
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def write(self, data: bytes):
        """
        向串口写入数据。

        Args:
            data (bytes): 要写入的数据。

        Returns:
            bytes: 实际写入的数据。
        """
        if isinstance(data, list):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        len_as_byte = len(data).to_bytes(1, self._byte_order)
        send_data = self._send_head + len_as_byte + data
        checksum = 0
        for i in range(0, len(send_data)):
            checksum += send_data[i]
            checksum &= 0xFF
        send_data += checksum.to_bytes(1, self._byte_order)
        self.ser.write(send_data)
        self.ser.flush()
        return send_data
