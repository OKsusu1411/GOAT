# generated from rosidl_generator_py/resource/_idl.py.em
# with input from imu_interface:msg/ImuState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_ImuState(type):
    """Metaclass of message 'ImuState'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('imu_interface')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'imu_interface.msg.ImuState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__imu_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__imu_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__imu_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__imu_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__imu_state

            from geometry_msgs.msg import Quaternion
            if Quaternion.__class__._TYPE_SUPPORT is None:
                Quaternion.__class__.__import_type_support__()

            from geometry_msgs.msg import Vector3
            if Vector3.__class__._TYPE_SUPPORT is None:
                Vector3.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class ImuState(metaclass=Metaclass_ImuState):
    """Message class 'ImuState'."""

    __slots__ = [
        '_header',
        '_quat',
        '_gyro',
        '_vel',
        '_mag',
        '_time_ms',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'quat': 'geometry_msgs/Quaternion',
        'gyro': 'geometry_msgs/Vector3',
        'vel': 'geometry_msgs/Vector3',
        'mag': 'geometry_msgs/Vector3',
        'time_ms': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Quaternion'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        from geometry_msgs.msg import Quaternion
        self.quat = kwargs.get('quat', Quaternion())
        from geometry_msgs.msg import Vector3
        self.gyro = kwargs.get('gyro', Vector3())
        from geometry_msgs.msg import Vector3
        self.vel = kwargs.get('vel', Vector3())
        from geometry_msgs.msg import Vector3
        self.mag = kwargs.get('mag', Vector3())
        self.time_ms = kwargs.get('time_ms', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.quat != other.quat:
            return False
        if self.gyro != other.gyro:
            return False
        if self.vel != other.vel:
            return False
        if self.mag != other.mag:
            return False
        if self.time_ms != other.time_ms:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def quat(self):
        """Message field 'quat'."""
        return self._quat

    @quat.setter
    def quat(self, value):
        if __debug__:
            from geometry_msgs.msg import Quaternion
            assert \
                isinstance(value, Quaternion), \
                "The 'quat' field must be a sub message of type 'Quaternion'"
        self._quat = value

    @builtins.property
    def gyro(self):
        """Message field 'gyro'."""
        return self._gyro

    @gyro.setter
    def gyro(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'gyro' field must be a sub message of type 'Vector3'"
        self._gyro = value

    @builtins.property
    def vel(self):
        """Message field 'vel'."""
        return self._vel

    @vel.setter
    def vel(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'vel' field must be a sub message of type 'Vector3'"
        self._vel = value

    @builtins.property
    def mag(self):
        """Message field 'mag'."""
        return self._mag

    @mag.setter
    def mag(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'mag' field must be a sub message of type 'Vector3'"
        self._mag = value

    @builtins.property
    def time_ms(self):
        """Message field 'time_ms'."""
        return self._time_ms

    @time_ms.setter
    def time_ms(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'time_ms' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'time_ms' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._time_ms = value
